// GLMB Track-Before-Detect (particle / SMC implementation) on GPU.
//
// Model (Papi, Vo, Vo, Fantacci, Beard, "Generalized Labeled Multi-Bernoulli
// Approximation of Multi-Object Densities", IEEE TSP 2015, Sec. V - TBD):
//   image z(u,v) = sum_targets h(u,v; x) + n(u,v),  n ~ N(0, sigma_n^2) i.i.d.
//   h = A * exp(-|p - x|^2 / (2 sigma_psf^2)),  SNR := A / sigma_n.
// Under the separable likelihood assumption (target footprints don't overlap)
// the GLMB is conjugate: every track l gets the factor
//   eta_l = \int p_l(x) g_l(z|x) dx,   g_l = prod_{pixels in T(x)} N(z;h,s)/N(z;0,s)
// and hypothesis weights become w(I) * prod_{l in I} eta_l.
//
// GPU: one CUDA block per track (NP particles, 1 particle / thread) fuses
//      predict -> patch likelihood ratio -> eta (log-sum-exp) -> state estimate
//      -> systematic resampling. A second stream runs the measurement-driven
//      birth proposal (separable matched filter + 5x5 NMS + threshold) over the
//      full frame. Host: delta-GLMB hypothesis bookkeeping (sample/merge/prune).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_87 glmb_tbd.cu -o glmb_tbd

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int W = 1920, H = 1080;
constexpr int NP = 1024;        // particles per track
constexpr int MAX_SLOTS = 2048; // max concurrent tracks on device
constexpr int PATCH_R = 4;      // likelihood patch radius (9x9)
constexpr int MF_R = 4;         // matched filter radius
constexpr int MAX_CAND = 8192;

struct Params {
  // sensor
  float sigma_n = 1.f, snr = 2.f, psf_sigma = 1.5f;
  // dynamics (CV, white-noise acceleration), units: px, frames
  float sigma_a = 0.15f;       // single-model (CV) process noise
  float pS = 0.99f;
  // IMM / jump-Markov multiple model: modes {CV, CT(+w0), CT(-w0)}
  bool imm = false;
  float omega0 = 0.08f;          // rad/frame
  float sa_cv = 0.05f, sa_ct = 0.15f;
  float p_stay_cv = 0.95f, p_stay_ct = 0.90f;
  bool maneuver = false;         // scenario with coordinated-turn segments
  // measurement-driven birth
  float rB = 0.01f, vmax = 4.f, birth_pos_std = 0.7f;
  float det_thresh = 4.0f, birth_gate = 8.f;
  int max_birth = 256;
  // GLMB management
  double r_prune = 1e-3;
  int H_max = 1000, Ns_total = 2000;
  float merge_dist = 3.f;
  // scenario
  int K = 100, n_targets = 20;
  unsigned seed = 2026;
};

constexpr int MAX_MODES = 3;
struct DevParams {
  float amp, inv_2s2, inv_sn2, vmax, bstd;
  int nmodes;
  float omega[MAX_MODES], sa[MAX_MODES];
  float tpm_cdf[MAX_MODES][MAX_MODES];  // row-wise CDF of the mode transition matrix
  float init_cdf[MAX_MODES];            // birth mode distribution
};
__constant__ DevParams cP;
__constant__ float cG[2 * MF_R + 1];

struct TruthPt { float x, y; };
struct SlotTask { int slot, birth; float bx, by; };
struct Cand { float x, y, score; };

// ---------------------------------------------------------------- scenario
__global__ void k_gen_frame(float* __restrict__ img, const TruthPt* __restrict__ t,
                            int nt, float sn, unsigned long long seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  int idx = y * W + x;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, idx, 0, &st);
  float v = sn * curand_normal(&st);
  for (int i = 0; i < nt; ++i) {
    float dx = x - t[i].x, dy = y - t[i].y, d2 = dx * dx + dy * dy;
    if (d2 < 100.f) v += cP.amp * __expf(-d2 * cP.inv_2s2);
  }
  img[idx] = v;
}

// ---------------------------------------------------------------- particle filter
struct Sum4 {
  __device__ float4 operator()(const float4& a, const float4& b) const {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
};
struct MaxF {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void __launch_bounds__(NP)
k_pf_step(const float* __restrict__ img, const float4* __restrict__ pin,
          float4* __restrict__ pout, const uint8_t* __restrict__ min,
          uint8_t* __restrict__ mout, const SlotTask* __restrict__ tasks,
          float* __restrict__ out_logeta, float4* __restrict__ out_mean,
          float* __restrict__ out_fin, float2* __restrict__ out_mode,
          unsigned long long seed, int frame) {
  typedef cub::BlockReduce<float, NP> BRf;
  typedef cub::BlockReduce<float4, NP> BR4;
  typedef cub::BlockScan<float, NP> BS;
  __shared__ union {
    typename BRf::TempStorage rf;
    typename BR4::TempStorage r4;
    typename BS::TempStorage s;
  } tmp;
  __shared__ float4 sp[NP];
  __shared__ float cdf[NP];
  __shared__ uint8_t sm[NP];
  __shared__ float bc[2];

  const SlotTask t = tasks[blockIdx.x];
  const int i = threadIdx.x;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, ((unsigned long long)frame * MAX_SLOTS + t.slot) * NP + i, 0, &st);

  // --- birth initialisation or load
  float4 p;
  int m = 0;
  if (t.birth) {
    float2 n = curand_normal2(&st);
    float2 u = make_float2(curand_uniform(&st), curand_uniform(&st));
    p = make_float4(t.bx + cP.bstd * n.x, t.by + cP.bstd * n.y,
                    cP.vmax * (2.f * u.x - 1.f), cP.vmax * (2.f * u.y - 1.f));
    float um = curand_uniform(&st);
    while (m < cP.nmodes - 1 && um > cP.init_cdf[m]) ++m;
  } else {
    p = pin[(size_t)t.slot * NP + i];
    m = min[(size_t)t.slot * NP + i];
    // --- IMM mode transition (Markov jump), sampled per particle
    float um = curand_uniform(&st);
    int mn = 0;
    while (mn < cP.nmodes - 1 && um > cP.tpm_cdf[m][mn]) ++mn;
    m = mn;
  }
  // --- mode-conditioned prediction: CV (omega=0) or coordinated turn (exact
  //     discretisation, known turn rate) + piecewise-constant white acceleration
  const float om = cP.omega[m];
  if (om != 0.f) {
    float sn, cs;
    __sincosf(om, &sn, &cs);
    float a1 = sn / om, a2 = (1.f - cs) / om;
    float nx = p.x + a1 * p.z - a2 * p.w, ny = p.y + a2 * p.z + a1 * p.w;
    float nvx = cs * p.z - sn * p.w, nvy = sn * p.z + cs * p.w;
    p = make_float4(nx, ny, nvx, nvy);
  } else {
    p.x += p.z; p.y += p.w;
  }
  float2 a = curand_normal2(&st);
  a.x *= cP.sa[m]; a.y *= cP.sa[m];
  p.x += 0.5f * a.x; p.y += 0.5f * a.y;
  p.z += a.x;        p.w += a.y;

  // --- log likelihood ratio over the target's pixel footprint
  const bool inside = p.x >= 0.f && p.x <= W - 1 && p.y >= 0.f && p.y <= H - 1;
  float ll = 0.f;
  if (inside) {
    int cx = __float2int_rn(p.x), cy = __float2int_rn(p.y);
#pragma unroll
    for (int dy = -PATCH_R; dy <= PATCH_R; ++dy) {
      int yy = cy + dy;
      if (yy < 0 || yy >= H) continue;
      float ddy = yy - p.y, ey = ddy * ddy;
#pragma unroll
      for (int dx = -PATCH_R; dx <= PATCH_R; ++dx) {
        int xx = cx + dx;
        if (xx < 0 || xx >= W) continue;
        float ddx = xx - p.x;
        float h = cP.amp * __expf(-(ddx * ddx + ey) * cP.inv_2s2);
        float z = __ldg(&img[yy * W + xx]);
        ll += h * (z - 0.5f * h);
      }
    }
    ll *= cP.inv_sn2;
  }

  // --- eta = mean_i exp(ll_i)  (particles are equally weighted after resampling)
  float mx = BRf(tmp.rf).Reduce(ll, MaxF());
  if (i == 0) bc[0] = mx;
  __syncthreads();
  mx = bc[0];
  float w = __expf(ll - mx);
  float4 s1 = BR4(tmp.r4).Reduce(make_float4(w, w * p.x, w * p.y, inside ? 1.f : 0.f), Sum4());
  __syncthreads();
  float4 s2 = BR4(tmp.r4).Reduce(
      make_float4(w * p.z, w * p.w, m == 1 ? w : 0.f, m == 2 ? w : 0.f), Sum4());
  __syncthreads();
  if (i == 0) {
    float sw = s1.x;
    out_logeta[blockIdx.x] = mx + logf(sw / NP);
    out_mean[blockIdx.x] = make_float4(s1.y / sw, s1.z / sw, s2.x / sw, s2.y / sw);
    out_fin[blockIdx.x] = s1.w / NP;
    out_mode[blockIdx.x] = make_float2(s2.z / sw, s2.w / sw);  // P(CT+), P(CT-)
    bc[1] = curand_uniform(&st);
  }

  // --- systematic resampling
  float c;
  BS(tmp.s).InclusiveSum(w, c);
  cdf[i] = c;
  sp[i] = p;
  sm[i] = (uint8_t)m;
  __syncthreads();
  const float total = cdf[NP - 1];
  const float u = ((float)i + bc[1]) * (total / NP);
  int lo = 0, hi = NP - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (cdf[mid] < u) lo = mid + 1; else hi = mid;
  }
  pout[(size_t)t.slot * NP + i] = sp[lo];
  mout[(size_t)t.slot * NP + i] = sm[lo];
}

// ---------------------------------------------------------------- birth proposal
__global__ void k_mf_rows(const float* __restrict__ in, float* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= W) return;
  float s = 0.f;
#pragma unroll
  for (int k = -MF_R; k <= MF_R; ++k) {
    int xx = x + k;
    if (xx >= 0 && xx < W) s += cG[k + MF_R] * __ldg(&in[y * W + xx]);
  }
  out[y * W + x] = s;
}
// output: matched filter statistic T = <h,z>/(sigma_n ||h||) ~ N(0,1) under H0
__global__ void k_mf_cols(const float* __restrict__ in, float* __restrict__ out, float norm) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= W) return;
  float s = 0.f;
#pragma unroll
  for (int k = -MF_R; k <= MF_R; ++k) {
    int yy = y + k;
    if (yy >= 0 && yy < H) s += cG[k + MF_R] * __ldg(&in[yy * W + x]);
  }
  out[y * W + x] = s * norm;
}
__global__ void k_nms(const float* __restrict__ T, float thr, Cand* __restrict__ cand,
                      int* __restrict__ ncand) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  float v = T[y * W + x];
  if (v < thr) return;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      if (!dx && !dy) continue;
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      float o = T[yy * W + xx];
      if (o > v || (o == v && (dy < 0 || (dy == 0 && dx < 0)))) return;
    }
  int k = atomicAdd(ncand, 1);
  if (k < MAX_CAND) cand[k] = {(float)x, (float)y, v};
}

// ---------------------------------------------------------------- host GLMB
struct VecHash {
  size_t operator()(const std::vector<int>& v) const {
    size_t h = 1469598103934665603ULL;
    for (int x : v) h = (h ^ (size_t)x) * 1099511628211ULL;
    return h;
  }
};
struct Hyp { std::vector<int> L; double lw; };
struct Track {
  int slot = -1, birth_k = 0;
  float4 mean{};
  float2 mode{};
  double la = 0, lb = 0, r = 0;
  bool live = false;
};

static double logaddexp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}
static void normalize(std::vector<Hyp>& hy) {
  double m = -INFINITY;
  for (auto& h : hy) m = std::max(m, h.lw);
  double s = 0;
  for (auto& h : hy) s += std::exp(h.lw - m);
  double ls = m + std::log(s);
  for (auto& h : hy) h.lw -= ls;
}
// merge identical label sets and re-normalise
static void merge_dupes(std::vector<Hyp>& hy) {
  std::unordered_map<std::vector<int>, size_t, VecHash> mp;
  std::vector<Hyp> out;
  out.reserve(hy.size());
  for (auto& h : hy) {
    auto it = mp.find(h.L);
    if (it == mp.end()) { mp.emplace(h.L, out.size()); out.push_back(std::move(h)); }
    else out[it->second].lw = logaddexp(out[it->second].lw, h.lw);
  }
  hy.swap(out);
  normalize(hy);
}

// ---------------------------------------------------------------- truth
struct TruthTarget { int tb, td; std::vector<float4> s; std::vector<int> mode; };  // states for frames [tb,td)

static std::vector<TruthTarget> gen_truth(const Params& P) {
  std::mt19937 rng(P.seed);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  std::normal_distribution<float> N01(0.f, 1.f);
  std::vector<TruthTarget> out;
  int tries = 0;
  while ((int)out.size() < P.n_targets && tries++ < 100000) {
    TruthTarget t;
    bool from_start = out.size() < (size_t)P.n_targets / 2;
    t.tb = from_start ? 0 : (int)(U(rng) * 50);
    t.td = (U(rng) < 0.5f) ? P.K : 60 + (int)(U(rng) * (P.K - 60));
    float x = 60 + U(rng) * (W - 120), y = 60 + U(rng) * (H - 120);
    float sp = P.maneuver ? 1.5f + 3.0f * U(rng) : 0.5f + 2.5f * U(rng);
    float hd = 6.2831853f * U(rng);
    float4 s = make_float4(x, y, sp * cosf(hd), sp * sinf(hd));
    int mode = 0, seg = 5 + (int)(U(rng) * 20);
    float om = 0.f;
    for (int k = t.tb; k < t.td; ++k) {
      if (k > t.tb) {
        if (P.maneuver && --seg <= 0) {  // alternate CV / coordinated-turn segments
          if (mode == 0) {
            mode = U(rng) < 0.5f ? 1 : 2;
            om = (mode == 1 ? 1.f : -1.f) * (0.05f + 0.07f * U(rng));
            seg = 8 + (int)(U(rng) * 17);
          } else { mode = 0; om = 0.f; seg = 10 + (int)(U(rng) * 20); }
        }
        if (om != 0.f) {
          float sn = sinf(om), cs = cosf(om), a1 = sn / om, a2 = (1 - cs) / om;
          s = make_float4(s.x + a1 * s.z - a2 * s.w, s.y + a2 * s.z + a1 * s.w,
                          cs * s.z - sn * s.w, sn * s.z + cs * s.w);
        } else { s.x += s.z; s.y += s.w; }
        float ax = 0.03f * N01(rng), ay = 0.03f * N01(rng);
        s.x += 0.5f * ax; s.y += 0.5f * ay; s.z += ax; s.w += ay;
      }
      if (s.x < 5 || s.y < 5 || s.x > W - 6 || s.y > H - 6) { t.td = k; break; }
      t.s.push_back(s);
      t.mode.push_back(mode);
    }
    if (t.td - t.tb < 15) continue;
    bool ok = true;  // keep footprints separated (separable likelihood holds)
    for (auto& o : out)
      for (int k = std::max(t.tb, o.tb); k < std::min(t.td, o.td) && ok; ++k) {
        float dx = t.s[k - t.tb].x - o.s[k - o.tb].x, dy = t.s[k - t.tb].y - o.s[k - o.tb].y;
        if (dx * dx + dy * dy < 20.f * 20.f) ok = false;
      }
    if (ok) out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  Params P;
  std::string outdir = "results";
  for (int a = 1; a < argc; ++a) {
    std::string s = argv[a];
    auto nxt = [&]() { return std::string(argv[++a]); };
    if (s == "--snr") P.snr = std::stof(nxt());
    else if (s == "--frames") P.K = std::stoi(nxt());
    else if (s == "--targets") P.n_targets = std::stoi(nxt());
    else if (s == "--seed") P.seed = std::stoul(nxt());
    else if (s == "--out") outdir = nxt();
    else if (s == "--thr") P.det_thresh = std::stof(nxt());
    else if (s == "--imm") P.imm = true;
    else if (s == "--maneuver") { P.maneuver = true; P.vmax = 6.f; }
    else if (s == "--sigma_a") P.sigma_a = std::stof(nxt());
    else if (s == "--vmax") P.vmax = std::stof(nxt());
    else if (s == "--omega") P.omega0 = std::stof(nxt());
  }
  std::string cmd = "mkdir -p " + outdir;
  if (system(cmd.c_str()) != 0) return 1;

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  printf("Image %dx%d, SNR=%.2f (A/sigma_n, %.2f dB), PSF sigma=%.2f px, %d particles/track\n",
         W, H, P.snr, 20 * log10f(P.snr), P.psf_sigma, NP);

  DevParams dp;
  dp.amp = P.snr * P.sigma_n;
  dp.inv_2s2 = 1.f / (2.f * P.psf_sigma * P.psf_sigma);
  dp.inv_sn2 = 1.f / (P.sigma_n * P.sigma_n);
  dp.vmax = P.vmax; dp.bstd = P.birth_pos_std;
  memset(dp.omega, 0, sizeof(dp.omega)); memset(dp.sa, 0, sizeof(dp.sa));
  if (P.imm) {
    dp.nmodes = 3;
    dp.omega[0] = 0.f; dp.omega[1] = P.omega0; dp.omega[2] = -P.omega0;
    dp.sa[0] = P.sa_cv; dp.sa[1] = dp.sa[2] = P.sa_ct;
    float T[3][3] = {{P.p_stay_cv, (1 - P.p_stay_cv) / 2, (1 - P.p_stay_cv) / 2},
                     {1 - P.p_stay_ct, P.p_stay_ct, 0.f},
                     {1 - P.p_stay_ct, 0.f, P.p_stay_ct}};
    for (int a = 0; a < 3; ++a) {
      float c = 0;
      for (int b = 0; b < 3; ++b) { c += T[a][b]; dp.tpm_cdf[a][b] = c; }
    }
    float ip[3] = {0.6f, 0.2f, 0.2f}, c = 0;
    for (int a = 0; a < 3; ++a) { c += ip[a]; dp.init_cdf[a] = c; }
    printf("IMM: modes CV(sa=%.2f), CT(+/-%.3f rad/frame, sa=%.2f), p_stay=%.2f/%.2f\n",
           P.sa_cv, P.omega0, P.sa_ct, P.p_stay_cv, P.p_stay_ct);
  } else {
    dp.nmodes = 1; dp.sa[0] = P.sigma_a;
    dp.tpm_cdf[0][0] = 1.f; dp.init_cdf[0] = 1.f;
    printf("Single model: CV(sa=%.2f)\n", P.sigma_a);
  }
  CK(cudaMemcpyToSymbol(cP, &dp, sizeof(dp)));
  float g[2 * MF_R + 1], g2 = 0;
  for (int k = -MF_R; k <= MF_R; ++k) {
    g[k + MF_R] = expf(-k * k * dp.inv_2s2);
    g2 += g[k + MF_R] * g[k + MF_R];
  }
  CK(cudaMemcpyToSymbol(cG, g, sizeof(g)));
  const float mf_norm = 1.f / (g2 * P.sigma_n);  // ||h||_2 = sum(g^2) for separable h

  // ---- truth
  auto truth = gen_truth(P);
  printf("Scenario: %zu targets, %d frames\n", truth.size(), P.K);
  {
    FILE* f = fopen((outdir + "/truth.csv").c_str(), "w");
    fprintf(f, "frame,id,x,y,vx,vy,mode\n");
    for (size_t i = 0; i < truth.size(); ++i)
      for (int k = truth[i].tb; k < truth[i].td; ++k) {
        auto s = truth[i].s[k - truth[i].tb];
        fprintf(f, "%d,%zu,%.3f,%.3f,%.3f,%.3f,%d\n", k, i, s.x, s.y, s.z, s.w,
                truth[i].mode[k - truth[i].tb]);
      }
    fclose(f);
  }

  // ---- device buffers
  const size_t npx = (size_t)W * H;
  float *d_gen, *d_img, *d_tmp, *d_T;
  float4 *d_pa, *d_pb, *d_mean;
  float *d_logeta, *d_fin;
  uint8_t *d_ma, *d_mb;
  float2* d_mode;
  SlotTask* d_tasks;
  TruthPt* d_truth;
  Cand* d_cand;
  int* d_ncand;
  CK(cudaMalloc(&d_gen, npx * 4)); CK(cudaMalloc(&d_img, npx * 4));
  CK(cudaMalloc(&d_tmp, npx * 4)); CK(cudaMalloc(&d_T, npx * 4));
  CK(cudaMalloc(&d_pa, (size_t)MAX_SLOTS * NP * sizeof(float4)));
  CK(cudaMalloc(&d_pb, (size_t)MAX_SLOTS * NP * sizeof(float4)));
  CK(cudaMalloc(&d_mean, MAX_SLOTS * sizeof(float4)));
  CK(cudaMalloc(&d_ma, (size_t)MAX_SLOTS * NP)); CK(cudaMalloc(&d_mb, (size_t)MAX_SLOTS * NP));
  CK(cudaMalloc(&d_mode, MAX_SLOTS * sizeof(float2)));
  CK(cudaMalloc(&d_logeta, MAX_SLOTS * 4)); CK(cudaMalloc(&d_fin, MAX_SLOTS * 4));
  CK(cudaMalloc(&d_tasks, MAX_SLOTS * sizeof(SlotTask)));
  CK(cudaMalloc(&d_truth, 256 * sizeof(TruthPt)));
  CK(cudaMalloc(&d_cand, MAX_CAND * sizeof(Cand)));
  CK(cudaMalloc(&d_ncand, sizeof(int)));

  float* h_frame;  // pinned "camera" buffer
  float2* h_mode;
  CK(cudaMallocHost(&h_mode, MAX_SLOTS * sizeof(float2)));
  SlotTask* h_tasks; float4* h_mean; float *h_logeta, *h_fin; Cand* h_cand; int* h_ncand;
  CK(cudaMallocHost(&h_frame, npx * 4));
  CK(cudaMallocHost(&h_tasks, MAX_SLOTS * sizeof(SlotTask)));
  CK(cudaMallocHost(&h_mean, MAX_SLOTS * sizeof(float4)));
  CK(cudaMallocHost(&h_logeta, MAX_SLOTS * 4)); CK(cudaMallocHost(&h_fin, MAX_SLOTS * 4));
  CK(cudaMallocHost(&h_cand, MAX_CAND * sizeof(Cand)));
  CK(cudaMallocHost(&h_ncand, sizeof(int)));

  cudaStream_t s_pf, s_det;
  CK(cudaStreamCreate(&s_pf)); CK(cudaStreamCreate(&s_det));
  cudaEvent_t e0, e_up, e_pf, e_det0, e_det;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e_up)); CK(cudaEventCreate(&e_pf));
  CK(cudaEventCreate(&e_det0)); CK(cudaEventCreate(&e_det));

  const dim3 b2(32, 8), g2d((W + 31) / 32, (H + 7) / 8);
  const dim3 brow(256), grow((W + 255) / 256, H);
  auto launch_detect = [&](cudaStream_t s) {
    CK(cudaMemsetAsync(d_ncand, 0, sizeof(int), s));
    k_mf_rows<<<grow, brow, 0, s>>>(d_img, d_tmp);
    k_mf_cols<<<grow, brow, 0, s>>>(d_tmp, d_T, mf_norm);
    k_nms<<<g2d, b2, 0, s>>>(d_T, P.det_thresh, d_cand, d_ncand);
    CK(cudaMemcpyAsync(h_ncand, d_ncand, sizeof(int), cudaMemcpyDeviceToHost, s));
  };

  // ---- warm-up (module load, cub init) - excluded from timing
  {
    k_gen_frame<<<g2d, b2>>>(d_img, d_truth, 0, P.sigma_n, 1);
    h_tasks[0] = {0, 1, 100.f, 100.f};
    CK(cudaMemcpy(d_tasks, h_tasks, sizeof(SlotTask), cudaMemcpyHostToDevice));
    for (int r = 0; r < 3; ++r) {
      k_pf_step<<<1, NP>>>(d_img, d_pa, d_pb, d_ma, d_mb, d_tasks, d_logeta, d_mean, d_fin,
                           d_mode, 1, 0);
      launch_detect(0);
    }
    CK(cudaDeviceSynchronize());
  }

  // ---- filter state
  std::vector<Track> tracks;  // indexed by track id
  std::vector<int> free_slots;
  for (int s = MAX_SLOTS - 1; s >= 0; --s) free_slots.push_back(s);
  std::vector<Hyp> hyps{{{}, 0.0}};
  std::vector<Cand> pending_births;  // proposals from frame k-1
  std::mt19937_64 rng(P.seed * 7919ULL + 1);
  std::uniform_real_distribution<double> U01(0.0, 1.0);

  FILE* fest = fopen((outdir + "/estimates.csv").c_str(), "w");
  fprintf(fest, "frame,label_k,label_id,x,y,vx,vy,r,p_cv,p_ctp,p_ctm\n");
  FILE* ftim = fopen((outdir + "/timing.csv").c_str(), "w");
  fprintf(ftim, "frame,total_ms,upload_ms,pf_ms,detect_ms,host_glmb_ms,gpu_pipeline_ms,n_tracks,n_births,n_cand,n_hyp,card_map\n");
  const int dump_frame = P.K / 2;
  float4* d_pcur = d_pa; float4* d_pnext = d_pb;
  uint8_t* d_mcur = d_ma; uint8_t* d_mnext = d_mb;

  for (int k = 0; k < P.K; ++k) {
    // ================= scenario: render frame k (NOT timed) =================
    std::vector<TruthPt> tp;
    for (auto& t : truth)
      if (k >= t.tb && k < t.td) tp.push_back({t.s[k - t.tb].x, t.s[k - t.tb].y});
    CK(cudaMemcpy(d_truth, tp.data(), tp.size() * sizeof(TruthPt), cudaMemcpyHostToDevice));
    k_gen_frame<<<g2d, b2>>>(d_gen, d_truth, (int)tp.size(), P.sigma_n,
                             (unsigned long long)P.seed * 1000003ULL + k);
    CK(cudaMemcpy(h_frame, d_gen, npx * 4, cudaMemcpyDeviceToHost));
    if (k == dump_frame) {
      FILE* f = fopen((outdir + "/frame.f32").c_str(), "wb");
      fwrite(h_frame, 4, npx, f);
      fclose(f);
    }
    CK(cudaDeviceSynchronize());

    // ================= tracker step (timed) =================
    auto t0 = std::chrono::high_resolution_clock::now();
    CK(cudaEventRecord(e0, s_pf));
    CK(cudaMemcpyAsync(d_img, h_frame, npx * 4, cudaMemcpyHostToDevice, s_pf));
    CK(cudaEventRecord(e_up, s_pf));

    // birth proposal on frame k runs concurrently with the particle filter
    CK(cudaStreamWaitEvent(s_det, e_up, 0));
    CK(cudaEventRecord(e_det0, s_det));
    launch_detect(s_det);
    CK(cudaEventRecord(e_det, s_det));

    // tasks: surviving tracks + births proposed from frame k-1
    std::vector<int> task_tid;
    int nt = 0;
    std::vector<char> in_hyp(tracks.size(), 0);
    for (auto& h : hyps) for (int l : h.L) in_hyp[l] = 1;
    for (size_t l = 0; l < tracks.size(); ++l)
      if (tracks[l].live && in_hyp[l]) {
        h_tasks[nt++] = {tracks[l].slot, 0, 0.f, 0.f};
        task_tid.push_back((int)l);
      }
    const int n_exist = nt;
    const int first_birth_id = (int)tracks.size();
    for (auto& c : pending_births) {
      if (free_slots.empty()) break;
      Track tr; tr.slot = free_slots.back(); free_slots.pop_back();
      tr.birth_k = k; tr.live = true;
      tracks.push_back(tr);
      h_tasks[nt++] = {tr.slot, 1, c.x, c.y};
      task_tid.push_back((int)tracks.size() - 1);
    }
    const int n_birth = nt - n_exist;
    if (nt > 0) {
      CK(cudaMemcpyAsync(d_tasks, h_tasks, nt * sizeof(SlotTask), cudaMemcpyHostToDevice, s_pf));
      k_pf_step<<<nt, NP, 0, s_pf>>>(d_img, d_pcur, d_pnext, d_mcur, d_mnext, d_tasks,
                                     d_logeta, d_mean, d_fin, d_mode, P.seed, k);
      CK(cudaMemcpyAsync(h_logeta, d_logeta, nt * 4, cudaMemcpyDeviceToHost, s_pf));
      CK(cudaMemcpyAsync(h_mean, d_mean, nt * sizeof(float4), cudaMemcpyDeviceToHost, s_pf));
      CK(cudaMemcpyAsync(h_fin, d_fin, nt * 4, cudaMemcpyDeviceToHost, s_pf));
      CK(cudaMemcpyAsync(h_mode, d_mode, nt * sizeof(float2), cudaMemcpyDeviceToHost, s_pf));
    }
    CK(cudaEventRecord(e_pf, s_pf));
    CK(cudaStreamSynchronize(s_pf));
    auto t_host0 = std::chrono::high_resolution_clock::now();

    // ---- GLMB update (joint prediction + TBD update, closed form under separability)
    for (int j = 0; j < nt; ++j) {
      Track& tr = tracks[task_tid[j]];
      tr.mean = h_mean[j];
      tr.mode = h_mode[j];
      double le = h_logeta[j];
      if (j < n_exist) {
        double ps = P.pS * std::max((double)h_fin[j], 1e-3);  // leaving FOV -> dies
        tr.la = std::log(ps) + le;
        tr.lb = std::log1p(-ps);
      } else {
        tr.la = std::log(P.rB) + le;
        tr.lb = std::log1p(-P.rB);
      }
    }
    std::vector<int> births(n_birth);
    std::iota(births.begin(), births.end(), first_birth_id);
    double birth_lb_sum = 0;
    for (int b : births) birth_lb_sum += tracks[b].lb;

    struct Child { double lw; int parent; };
    std::unordered_map<std::vector<int>, Child, VecHash> children;
    children.reserve(4 * P.Ns_total);
    for (size_t pi = 0; pi < hyps.size(); ++pi) {
      const Hyp& par = hyps[pi];
      std::vector<int> C = par.L;
      C.insert(C.end(), births.begin(), births.end());  // sorted: birth ids are largest
      double base = par.lw;
      for (int l : par.L) base += tracks[l].lb;
      base += birth_lb_sum;
      auto add = [&](const std::vector<char>& inc) {
        std::vector<int> J;
        double lw = base;
        for (size_t c = 0; c < C.size(); ++c)
          if (inc[c]) { J.push_back(C[c]); lw += tracks[C[c]].la - tracks[C[c]].lb; }
        auto it = children.find(J);
        if (it == children.end()) children.emplace(std::move(J), Child{lw, (int)pi});
        else if (it->second.parent != (int)pi) {
          it->second.lw = logaddexp(it->second.lw, lw);
          it->second.parent = (int)pi;
        }
      };
      std::vector<char> inc(C.size());
      std::vector<double> q(C.size());
      for (size_t c = 0; c < C.size(); ++c) {
        double d = tracks[C[c]].la - tracks[C[c]].lb;
        q[c] = 1.0 / (1.0 + std::exp(-d));
        inc[c] = d > 0;
      }
      add(inc);  // most probable child
      int ns = (int)std::ceil(P.Ns_total * std::exp(par.lw));
      for (int s = 0; s < ns; ++s) {  // children factorise -> exact independent draws
        for (size_t c = 0; c < C.size(); ++c) inc[c] = U01(rng) < q[c];
        add(inc);
      }
    }
    std::vector<Hyp> nh;
    nh.reserve(children.size());
    for (auto& kv : children) nh.push_back({kv.first, kv.second.lw});
    normalize(nh);
    std::sort(nh.begin(), nh.end(), [](const Hyp& a, const Hyp& b) { return a.lw > b.lw; });
    if ((int)nh.size() > P.H_max) nh.resize(P.H_max);
    normalize(nh);

    // ---- marginal existence, label pruning
    auto marginals = [&]() {
      for (auto& t : tracks) t.r = 0;
      for (auto& h : nh) { double w = std::exp(h.lw); for (int l : h.L) tracks[l].r += w; }
    };
    marginals();
    {
      std::vector<char> kill(tracks.size(), 0);
      bool any = false;
      for (size_t l = 0; l < tracks.size(); ++l)
        if (tracks[l].live && tracks[l].r < P.r_prune) { kill[l] = 1; any = true; }
      if (any) {
        std::vector<Hyp> kept;
        for (auto& h : nh) {
          bool ok = true;
          for (int l : h.L) if (kill[l]) { ok = false; break; }
          if (ok) kept.push_back(std::move(h));
        }
        if (kept.empty()) kept.push_back({{}, 0.0});
        nh.swap(kept);
        normalize(nh);
      }
    }
    // ---- merge duplicate tracks locked on the same target (separability guard)
    {
      std::vector<int> conf;
      marginals();
      for (size_t l = 0; l < tracks.size(); ++l) if (tracks[l].r > 0.5) conf.push_back((int)l);
      std::vector<int> repl(tracks.size());
      std::iota(repl.begin(), repl.end(), 0);
      bool any = false;
      for (size_t a = 0; a < conf.size(); ++a)
        for (size_t b = a + 1; b < conf.size(); ++b) {
          auto& A = tracks[conf[a]]; auto& B = tracks[conf[b]];
          float dx = A.mean.x - B.mean.x, dy = A.mean.y - B.mean.y;
          if (dx * dx + dy * dy < P.merge_dist * P.merge_dist && repl[conf[b]] == conf[b]) {
            repl[conf[b]] = conf[a];  // keep older label
            any = true;
          }
        }
      if (any) {
        for (auto& h : nh) {
          for (int& l : h.L) l = repl[l];
          std::sort(h.L.begin(), h.L.end());
          h.L.erase(std::unique(h.L.begin(), h.L.end()), h.L.end());
        }
        merge_dupes(nh);
      }
    }
    hyps.swap(nh);
    marginals();
    // free slots of labels that disappeared
    {
      std::vector<char> used(tracks.size(), 0);
      for (auto& h : hyps) for (int l : h.L) used[l] = 1;
      for (size_t l = 0; l < tracks.size(); ++l)
        if (tracks[l].live && !used[l]) { tracks[l].live = false; free_slots.push_back(tracks[l].slot); }
    }

    // ---- estimate: MAP cardinality, then best hypothesis with that cardinality
    std::vector<double> card(64, 0.0);
    for (auto& h : hyps) if (h.L.size() < card.size()) card[h.L.size()] += std::exp(h.lw);
    int nmap = (int)(std::max_element(card.begin(), card.end()) - card.begin());
    const Hyp* best = nullptr;
    for (auto& h : hyps) if ((int)h.L.size() == nmap && (!best || h.lw > best->lw)) best = &h;
    if (best)
      for (int l : best->L) {
        auto& m = tracks[l].mean;
        auto& md = tracks[l].mode;
        fprintf(fest, "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n", k, tracks[l].birth_k,
                l, m.x, m.y, m.z, m.w, tracks[l].r, 1.f - md.x - md.y, md.x, md.y);
      }

    // ---- births for k+1: detections on frame k, gated away from existing tracks
    CK(cudaStreamSynchronize(s_det));
    int ncand = std::min(*h_ncand, MAX_CAND);
    CK(cudaMemcpy(h_cand, d_cand, ncand * sizeof(Cand), cudaMemcpyDeviceToHost));
    pending_births.clear();
    for (int c = 0; c < ncand; ++c) {
      bool ok = true;
      for (auto& t : tracks)
        if (t.live) {
          float dx = t.mean.x - h_cand[c].x, dy = t.mean.y - h_cand[c].y;
          if (dx * dx + dy * dy < P.birth_gate * P.birth_gate) { ok = false; break; }
        }
      if (ok) pending_births.push_back(h_cand[c]);
    }
    std::sort(pending_births.begin(), pending_births.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    if ((int)pending_births.size() > P.max_birth) pending_births.resize(P.max_birth);
    std::swap(d_pcur, d_pnext);
    std::swap(d_mcur, d_mnext);

    auto t1 = std::chrono::high_resolution_clock::now();
    float up_ms, pf_ms, det_ms, gpu_ms;
    CK(cudaEventElapsedTime(&up_ms, e0, e_up));
    CK(cudaEventElapsedTime(&pf_ms, e_up, e_pf));
    CK(cudaEventElapsedTime(&det_ms, e_det0, e_det));
    float det_end_ms;
    CK(cudaEventElapsedTime(&det_end_ms, e0, e_det));
    gpu_ms = std::max(up_ms + pf_ms, det_end_ms);
    double tot = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double hst = std::chrono::duration<double, std::milli>(t1 - t_host0).count();
    int nlive = 0;
    for (auto& t : tracks) nlive += t.live;
    fprintf(ftim, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%zu,%d\n", k, tot, up_ms, pf_ms,
            det_ms, hst, gpu_ms, nt, n_birth, ncand, hyps.size(), nmap);
    if (k % 10 == 0 || k == P.K - 1)
      printf("k=%3d  tracks=%4d (births %3d) cand=%3d hyps=%4zu  |X|_MAP=%2d  live=%4d  %.2f ms\n",
             k, nt, n_birth, ncand, hyps.size(), nmap, nlive, tot);
  }
  fclose(fest);
  fclose(ftim);
  printf("Wrote %s/{truth,estimates,timing}.csv\n", outdir.c_str());
  return 0;
}

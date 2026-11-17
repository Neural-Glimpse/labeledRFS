# GLMB Track-Before-Detect on GPU (CUDA)

Particle GLMB-TBD (Papi, Vo, Vo, Fantacci, Beard, IEEE TSP 2015) under the separable image likelihood:
for each track l the update factor is `eta_l = ∫ p_l(x) g_l(z|x) dx`, and hypothesis weights become `w(I)·∏ eta_l`.

* `k_pf_step` – one CUDA block per track (1024 particles) does predict, 9x9 patch log-likelihood ratio, eta (log-sum-exp), state estimate and systematic resampling in a single kernel.
* `k_mf_rows/k_mf_cols/k_nms` – full-frame matched filter plus 5x5 NMS. These propose births for frame k+1. They run on a second stream at the same time as the particle filter.
* Host – delta-GLMB joint prediction/update. Children factorise, so they are sampled exactly. Then merge, truncate (H_max = 1000), prune labels with r < 1e-3, and estimate with the MAP cardinality.

Synthetic data: 1920x1080 float frames with i.i.d. N(0,1) noise. Each target is a Gaussian PSF (sigma 1.5 px) with peak amplitude A, and SNR = A/sigma_n (default 2, i.e. 6 dB).

```
nvcc -O3 -std=c++17 -arch=sm_87 glmb_tbd.cu -o glmb_tbd
./glmb_tbd --snr 2 --frames 100 --targets 20 --seed 2026 --out results
python3 evaluate.py results      # metrics.json, summary.png, frame.png
```

## IMM (jump-Markov multiple-model PF)

`--imm` gives each particle a mode index (CV, CT(+w0) or CT(-w0)). Before each prediction the mode is
re-sampled from a Markov transition matrix (p_stay 0.95 for CV and 0.90 for CT). The particle is then
propagated with that mode's dynamics. Per-track mode probabilities are the weighted particle fractions,
written to `estimates.csv` as `p_cv, p_ctp, p_ctm`. The GLMB update is unchanged: eta is integrated over
the mode-augmented state.

`--maneuver` generates targets that alternate between CV and coordinated-turn segments
(|w| in 0.05-0.12 rad/frame, speed 1.5-4.5 px/frame).

```
./glmb_tbd --maneuver --imm --out res_imm && python3 evaluate.py res_imm
./glmb_tbd --maneuver --sigma_a 0.4 --out res_cv && python3 evaluate.py res_cv
```

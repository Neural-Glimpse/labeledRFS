"""Metrics + plots for the GPU GLMB-TBD run.

usage: python3 evaluate.py [results_dir]
"""
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 1920, 1080
C_OSPA, P_OSPA = 10.0, 1      # OSPA cut-off (px) and order
GATE = 5.0                    # px, gate for CLEAR-MOT matching
OSPA2_WIN = 10                # frames, OSPA^(2) sliding window


def load(d):
    tr = np.genfromtxt(f"{d}/truth.csv", delimiter=",", names=True)
    es = np.genfromtxt(f"{d}/estimates.csv", delimiter=",", names=True)
    tm = np.genfromtxt(f"{d}/timing.csv", delimiter=",", names=True)
    es = np.atleast_1d(es)
    return tr, es, tm


def ospa(X, Y, c=C_OSPA, p=P_OSPA):
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m == 0 or n == 0:
        return c, 0.0, c
    D = np.minimum(np.linalg.norm(X[:, None] - Y[None], axis=2), c) ** p
    r, s = linear_sum_assignment(D)
    cost = D[r, s].sum()
    nmax = max(m, n)
    loc = (cost / nmax) ** (1 / p)
    card = ((c ** p) * abs(m - n) / nmax) ** (1 / p)
    return ((cost + c ** p * abs(m - n)) / nmax) ** (1 / p), loc, card


def ospa2(tracks_x, tracks_y, frames, c=C_OSPA, p=P_OSPA):
    """OSPA^(2) (Beard et al. 2020) over a set of frames. tracks: dict id -> {frame: xy}."""
    fs = set(frames)
    X = [t for t in tracks_x.values() if fs & t.keys()]
    Y = [t for t in tracks_y.values() if fs & t.keys()]
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return c
    D = np.zeros((m, n))
    for i, a in enumerate(X):
        for j, b in enumerate(Y):
            ts = (a.keys() | b.keys()) & fs
            s = 0.0
            for t in ts:
                if t in a and t in b:
                    s += min(c, np.linalg.norm(a[t] - b[t])) ** p
                else:
                    s += c ** p
            D[i, j] = (s / len(ts)) ** (1 / p)
    D = D ** p
    r, s = linear_sum_assignment(D)
    return ((D[r, s].sum() + c ** p * abs(m - n)) / max(m, n)) ** (1 / p)


def main(d="results"):
    tr, es, tm = load(d)
    K = int(tm["frame"].max()) + 1
    T = {k: (tr["id"][tr["frame"] == k].astype(int), np.c_[tr["x"], tr["y"]][tr["frame"] == k]) for k in range(K)}
    E = {k: (es["label_id"][es["frame"] == k].astype(int), np.c_[es["x"], es["y"]][es["frame"] == k]) for k in range(K)}

    # ---------------- per-frame OSPA, cardinality, CLEAR-MOT
    o, ol, oc, nt, ne = [], [], [], [], []
    TP = FP = FN = IDSW = 0
    dsum = 0.0
    last = {}
    first_hit = {}
    for k in range(K):
        ti, tx = T[k]
        ei, ex = E[k]
        a, b, c = ospa(tx, ex)
        o.append(a); ol.append(b); oc.append(c); nt.append(len(tx)); ne.append(len(ex))
        matched_t = set()
        if len(tx) and len(ex):
            D = np.linalg.norm(tx[:, None] - ex[None], axis=2)
            r, s = linear_sum_assignment(D)
            for i, j in zip(r, s):
                if D[i, j] <= GATE:
                    TP += 1; dsum += D[i, j]; matched_t.add(i)
                    gid, lab = ti[i], ei[j]
                    if gid in last and last[gid] != lab:
                        IDSW += 1
                    last[gid] = lab
                    first_hit.setdefault(gid, k)
        FN += len(tx) - len(matched_t)
        FP += len(ex) - len(matched_t)
    GT = sum(nt)
    births = {int(g): int(tr["frame"][tr["id"] == g].min()) for g in np.unique(tr["id"])}
    delay = [first_hit[g] - births[g] for g in births if g in first_hit]

    # ---------------- OSPA^(2)
    tx_tracks, ey_tracks = {}, {}
    for f, g, x, y in zip(tr["frame"], tr["id"], tr["x"], tr["y"]):
        tx_tracks.setdefault(int(g), {})[int(f)] = np.array([x, y])
    for f, g, x, y in zip(es["frame"], es["label_id"], es["x"], es["y"]):
        ey_tracks.setdefault(int(g), {})[int(f)] = np.array([x, y])
    o2_win = [ospa2(tx_tracks, ey_tracks, range(max(0, k - OSPA2_WIN + 1), k + 1)) for k in range(K)]
    o2_all = ospa2(tx_tracks, ey_tracks, range(K))

    # ---------------- latency (frame 0 excluded: empty filter)
    lat = {}
    for col in ["total_ms", "upload_ms", "pf_ms", "detect_ms", "host_glmb_ms", "gpu_pipeline_ms"]:
        v = tm[col][1:]
        lat[col] = dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                        p99=float(np.percentile(v, 99)), max=float(v.max()))
    comp = tm["total_ms"][1:] - tm["upload_ms"][1:]

    m = dict(
        frames=K, gt_objects=int(GT), gt_targets=len(births),
        ospa_mean=float(np.mean(o)), ospa_loc_mean=float(np.mean(ol)), ospa_card_mean=float(np.mean(oc)),
        ospa_mean_after_k5=float(np.mean(o[5:])),
        ospa2_window10_mean=float(np.mean(o2_win)), ospa2_full=float(o2_all),
        card_abs_err_mean=float(np.mean(np.abs(np.array(ne) - np.array(nt)))),
        card_bias=float(np.mean(np.array(ne) - np.array(nt))),
        TP=TP, FP=FP, FN=FN, IDSW=IDSW,
        recall=TP / GT, precision=TP / max(TP + FP, 1),
        MOTA=1 - (FN + FP + IDSW) / GT, MOTP_px=dsum / max(TP, 1),
        rmse_matched_px=None,
        n_est_labels=len(ey_tracks),
        confirm_delay_frames_mean=float(np.mean(delay)) if delay else None,
        confirm_delay_frames_max=int(np.max(delay)) if delay else None,
        latency_ms=lat,
        compute_only_ms=dict(mean=float(comp.mean()), p50=float(np.median(comp)),
                             p95=float(np.percentile(comp, 95)), max=float(comp.max())),
        fps_mean=1000.0 / lat["total_ms"]["mean"],
        tracks_processed_mean=float(tm["n_tracks"][1:].mean()),
        births_per_frame_mean=float(tm["n_births"][1:].mean()),
        hyps_mean=float(tm["n_hyp"][1:].mean()),
    )
    # RMSE over gated matches
    sq, cnt = 0.0, 0
    for k in range(K):
        _, tx = T[k]; _, ex = E[k]
        if len(tx) and len(ex):
            D = np.linalg.norm(tx[:, None] - ex[None], axis=2)
            r, s = linear_sum_assignment(D)
            dd = D[r, s]; dd = dd[dd <= GATE]
            sq += (dd ** 2).sum(); cnt += len(dd)
    m["rmse_matched_px"] = float(np.sqrt(sq / max(cnt, 1)))

    # per-truth-mode breakdown (CV vs coordinated turn) and IMM mode accuracy
    if "mode" in tr.dtype.names:
        has_p = "p_cv" in es.dtype.names
        stats = {0: [0, 0, 0.0], 1: [0, 0, 0.0]}  # gt count, hits, sq err  (1 = turning)
        conf = np.zeros((3, 3), int)
        for k in range(K):
            sel_t = tr["frame"] == k; sel_e = es["frame"] == k
            tx = np.c_[tr["x"], tr["y"]][sel_t]; tm_ = tr["mode"][sel_t].astype(int)
            ex = np.c_[es["x"], es["y"]][sel_e]
            pm = np.c_[es["p_cv"], es["p_ctp"], es["p_ctm"]][sel_e] if has_p else None
            for md in tm_:
                stats[int(md > 0)][0] += 1
            if len(tx) and len(ex):
                D = np.linalg.norm(tx[:, None] - ex[None], axis=2)
                r, s = linear_sum_assignment(D)
                for i, j in zip(r, s):
                    if D[i, j] <= GATE:
                        b = int(tm_[i] > 0)
                        stats[b][1] += 1; stats[b][2] += D[i, j] ** 2
                        if has_p:
                            conf[tm_[i], int(np.argmax(pm[j]))] += 1
        for b, name in [(0, "cv"), (1, "turn")]:
            n, h, e = stats[b]
            m[f"recall_{name}"] = h / max(n, 1)
            m[f"rmse_{name}_px"] = float(np.sqrt(e / max(h, 1)))
            m[f"gt_{name}"] = n
        if has_p and conf.sum():
            m["mode_accuracy"] = float(np.trace(conf) / conf.sum())
            m["mode_confusion_rows_truth_cv_ctp_ctm"] = conf.tolist()

    with open(f"{d}/metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))

    # ---------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ks = np.arange(K)
    ax[0, 0].plot(ks, nt, "k-", label="truth"); ax[0, 0].plot(ks, ne, "C0.", label="estimate (MAP)")
    ax[0, 0].set_title("Cardinality"); ax[0, 0].set_xlabel("frame"); ax[0, 0].legend()
    ax[0, 1].plot(ks, o, label="OSPA"); ax[0, 1].plot(ks, ol, label="loc"); ax[0, 1].plot(ks, oc, label="card")
    ax[0, 1].plot(ks, o2_win, label=f"OSPA$^{{(2)}}$ (win {OSPA2_WIN})")
    ax[0, 1].set_title(f"OSPA (c={C_OSPA:g}px, p={P_OSPA})"); ax[0, 1].set_xlabel("frame"); ax[0, 1].legend()
    ax[1, 0].plot(ks[1:], tm["total_ms"][1:], label="end-to-end (incl. 8 MB upload)")
    ax[1, 0].plot(ks[1:], tm["pf_ms"][1:], label="PF kernel (GPU)")
    ax[1, 0].plot(ks[1:], tm["detect_ms"][1:], label="birth proposal (GPU)")
    ax[1, 0].plot(ks[1:], tm["upload_ms"][1:], label="H2D upload")
    ax[1, 0].plot(ks[1:], tm["host_glmb_ms"][1:], label="host GLMB")
    ax[1, 0].set_title("Latency per frame"); ax[1, 0].set_ylabel("ms"); ax[1, 0].set_xlabel("frame")
    ax[1, 0].legend(fontsize=8)
    for g, t in tx_tracks.items():
        f = sorted(t); p = np.array([t[i] for i in f])
        ax[1, 1].plot(p[:, 0], p[:, 1], "k-", lw=3, alpha=0.3)
    for g, t in ey_tracks.items():
        f = sorted(t); p = np.array([t[i] for i in f])
        ax[1, 1].plot(p[:, 0], p[:, 1], ".", ms=2)
    ax[1, 1].set_xlim(0, W); ax[1, 1].set_ylim(H, 0); ax[1, 1].set_aspect("equal")
    ax[1, 1].set_title("Tracks (grey: truth, colour: GLMB labels)")
    fig.tight_layout(); fig.savefig(f"{d}/summary.png", dpi=110)

    # frame crop around a target
    try:
        img = np.fromfile(f"{d}/frame.f32", dtype=np.float32).reshape(H, W)
        kd = K // 2
        _, tx = T[kd]; _, ex = E[kd]
        cx, cy = tx[0]
        x0, y0 = int(max(0, cx - 60)), int(max(0, cy - 60))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].imshow(img, cmap="gray", vmin=-3, vmax=5); ax[0].set_title(f"Full frame {kd} (SNR=2)")
        ax[0].plot(tx[:, 0], tx[:, 1], "go", mfc="none", ms=10)
        ax[1].imshow(img[y0:y0 + 120, x0:x0 + 120], cmap="gray", vmin=-3, vmax=5,
                     extent=(x0 - .5, x0 + 119.5, y0 + 119.5, y0 - .5))
        ax[1].plot(tx[:, 0], tx[:, 1], "go", mfc="none", ms=16, label="truth")
        ax[1].plot(ex[:, 0], ex[:, 1], "r+", ms=14, label="GLMB estimate")
        ax[1].set_xlim(x0 - .5, x0 + 119.5); ax[1].set_ylim(y0 + 119.5, y0 - .5)
        ax[1].legend(); ax[1].set_title("120x120 crop")
        fig.tight_layout(); fig.savefig(f"{d}/frame.png", dpi=110)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

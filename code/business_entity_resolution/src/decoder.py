"""Set decoders for per-S1 predictions and the fold-level metric / error budget (pure NumPy, no I/O).

Pairs are described by aligned arrays: ``ent`` (entity index 0..n-1 of the pair's S1, -1 = outside the evaluated
set) and a probability ``p``. One-to-one is applied before decoding (each query keeps its argmax S1).

expected_f05_decode: per S1, sort its kept candidates by p and choose the prefix (including the EMPTY set) that
maximises the exact expected F0.5 under independent Bernoulli(p) pair labels, plus Poisson(lambda) true matches
lost in blocking (lambda = miss / (1 - miss) x the expected number of matches found). With an S1-level
P(has any match) = h, the S1 is a singleton with probability 1 - h and otherwise its pairs are Bernoulli(min(p / h, 1)):
    E[F | set] = (1 - h) [set empty] + h E_Bernoulli[F | set].
"""
import numpy as np
import pandas as pd

BETA2 = 0.25          # F0.5: beta^2
K_MAX = 12            # candidates per S1 considered by the decoder (after one-to-one, sorted by p)
M_MAX = 4             # Poisson truncation for matches lost in blocking


def logit_temperature(p: np.ndarray, t: float) -> np.ndarray:
    """Return sigmoid(logit(p) / t) (t > 1 flattens, t < 1 sharpens)."""
    if t == 1.0:
        return p
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(q / (1 - q)) / t))


def padded(ent: np.ndarray, p: np.ndarray, n: int, k_max: int = K_MAX) -> tuple:
    """Return (P n x k_max probabilities sorted desc per entity, pair index matrix (-1 = pad), dropped count)."""
    m = ent >= 0
    idx = np.flatnonzero(m)
    order = idx[np.lexsort((-p[idx], ent[idx]))]
    e = ent[order]
    start = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1]
    pos = np.arange(len(e)) - np.repeat(start, np.diff(np.r_[start, len(e)]))
    ok = pos < k_max
    P = np.zeros((n, k_max))
    I = np.full((n, k_max), -1, dtype=np.int64)
    P[e[ok], pos[ok]] = p[order[ok]]
    I[e[ok], pos[ok]] = order[ok]
    return P, I, int((~ok).sum())


def _poisson(lam: np.ndarray, m_max: int = M_MAX) -> np.ndarray:
    """Return the truncated (renormalised) Poisson pmf 0..m_max for each lambda (n x (m_max + 1))."""
    k = np.arange(m_max + 1)
    fact = np.array([np.prod(np.arange(1, i + 1)) for i in k], dtype=float)
    pmf = np.exp(-lam[:, None]) * lam[:, None] ** k / fact
    return pmf / pmf.sum(1, keepdims=True)


def expected_f05_table(P: np.ndarray, h: np.ndarray = None, miss: float = 0.0) -> np.ndarray:
    """Return E[F0.5] for every prefix size k = 0..K of each row of sorted probabilities P (n x (K + 1))."""
    n, K = P.shape
    if h is not None:
        h = np.clip(h, 1e-6, 1.0)
        Q = np.minimum(P / h[:, None], 1.0)
    else:
        Q = P
    lam = miss / (1 - miss) * Q.sum(1) if miss > 0 else np.zeros(n)
    M = _poisson(lam)
    # backward: distribution of the number of true pairs among items k..K-1, convolved with the missed count
    B = np.zeros((K + 1, n, K + M_MAX + 1))
    B[K, :, :M_MAX + 1] = M
    for k in range(K - 1, -1, -1):
        q = Q[:, k:k + 1]
        B[k] = B[k + 1] * (1 - q)
        B[k, :, 1:] += B[k + 1, :, :-1] * q
    E = np.zeros((n, K + 1))
    E[:, 0] = B[0, :, 0]                                   # empty set is right iff there is no true match at all
    D = np.zeros((n, K + 1))
    D[:, 0] = 1.0                                          # forward: TP count among the first k items
    a = np.arange(K + 1)[:, None]
    b = np.arange(K + M_MAX + 1)[None, :]
    for k in range(1, K + 1):
        q = Q[:, k - 1:k]
        D = D * (1 - q) + np.concatenate([np.zeros((n, 1)), D[:, :-1]], 1) * q
        W = np.where(a > 0, (1 + BETA2) * a / (k + BETA2 * (a + b)), 0.0)      # (K+1) x (K+M+1)
        E[:, k] = np.einsum("na,nb,ab->n", D, B[k], W)
    if h is not None:
        E = h[:, None] * E
        E[:, 0] += 1 - h
    return E


def expected_f05_decode(ent: np.ndarray, p: np.ndarray, n: int, h: np.ndarray = None, temperature: float = 1.0,
                        miss: float = 0.0, chunk: int = 100_000) -> np.ndarray:
    """Return the predicted-pair mask choosing, per entity, the prefix with the highest expected F0.5.

    ``ent``/``p`` should already be restricted to one-to-one kept pairs (others: ent = -1).
    """
    P, I, _ = padded(ent, logit_temperature(p, temperature), n)
    keep = np.zeros(len(p), dtype=bool)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        E = expected_f05_table(P[sl], None if h is None else h[sl], miss)
        best = E.argmax(1)
        take = np.arange(P.shape[1])[None, :] < best[:, None]
        rows = I[sl][take & (I[sl] >= 0)]
        keep[rows] = True
    return keep


def threshold_decode(ent: np.ndarray, p: np.ndarray, t_accept: float, t_keep: float) -> np.ndarray:
    """Return the mask p >= t_accept, keeping an S1's pairs only if its best p >= t_keep (the empty rule)."""
    keep = (ent >= 0) & (p >= t_accept)
    if t_keep > t_accept:
        mx = pd.Series(np.where(keep, p, -np.inf)).groupby(ent).transform("max").to_numpy()
        keep &= mx >= t_keep
    return keep


# ------------------------------------------------------------------ metric + error budget
def entity_counts(ent: np.ndarray, pred: np.ndarray, label: np.ndarray, n: int) -> tuple:
    """Return (predicted, true-positive) counts per entity for the predicted pairs."""
    m = pred & (ent >= 0)
    return (np.bincount(ent[m], minlength=n).astype(float),
            np.bincount(ent[m], weights=label[m].astype(float), minlength=n))


def f05_per_entity(k: np.ndarray, tp: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Return F0.5 per entity (singleton: 1 if empty else 0; empty prediction on a non-singleton: 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(k == 0, 0.0, (1 + BETA2) * tp / (k + BETA2 * n_true))
    return np.where(n_true == 0, (k == 0).astype(float), f)


def fold_metrics(ent, pred, label, n_true) -> dict:
    """Return macro F0.5 and diagnostics for a predicted-pair mask over entities 0..len(n_true)-1."""
    n = len(n_true)
    k, tp = entity_counts(ent, pred, label, n)
    f = f05_per_entity(k, tp, n_true)
    single = n_true == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(k == 0, 1.0, tp / np.maximum(k, 1))
        rec = np.where(single, 1.0, tp / np.maximum(n_true, 1))
    return {"macro_f05": float(f.mean()), "precision": float(prec.mean()), "recall": float(rec.mean()),
            "pred_per_s1": float(k.mean()), "pct_empty": 100 * float((k == 0).mean()),
            "singleton_share": float(single.mean()),
            "f05_singletons": float(f[single].mean()) if single.any() else 1.0,
            "f05_non_singletons": float(f[~single].mean()) if (~single).any() else 1.0, "n_entities": int(n)}


def error_budget(ent, pred, label, q_true_s1, n_true) -> dict:
    """Split (1 - macro F0.5) into points lost to: singleton predicted non-empty; false positives on non-singletons
    (decoy = the query matches nobody / other = it belongs to another S1); false negatives missed by blocking (true
    match absent from the candidate table) / by scoring or the decoder (present but not predicted).

    Non-singleton loss 1 - F is attributed sequentially: removing the false positives gives F_noFP
    (1.25 tp / (tp + 0.25 N), 0 if tp = 0); FP loss = F_noFP - F (split decoy / other by count), FN loss = 1 - F_noFP
    (split blocking / scoring by count). Values are points of macro F0.5 (they sum to 1 - F0.5).
    """
    n = len(n_true)
    m = pred & (ent >= 0)
    k, tp = entity_counts(ent, pred, label, n)
    fp_dec = np.bincount(ent[m], weights=((label[m] == 0) & (q_true_s1[m] < 0)).astype(float), minlength=n)
    fp_oth = k - tp - fp_dec
    inb = ent >= 0
    found = np.bincount(ent[inb], weights=label[inb].astype(float), minlength=n)
    f = f05_per_entity(k, tp, n_true)
    single = n_true == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_nofp = np.where(tp > 0, (1 + BETA2) * tp / (tp + BETA2 * n_true), 0.0)
        loss_fp = np.where(single, 0.0, f_nofp - f)
        loss_fn = np.where(single, 0.0, 1 - f_nofp)
        dec_share = np.where(fp_dec + fp_oth > 0, fp_dec / (fp_dec + fp_oth), 0.0)
        miss_block = np.maximum(n_true - found, 0)
        miss_score = np.maximum(found - tp, 0)
        blk_share = np.where(miss_block + miss_score > 0, miss_block / (miss_block + miss_score), 0.0)
    out = {"singleton_nonempty": float((single & (k > 0)).sum() / n),
           "fp_decoy": float((loss_fp * dec_share).sum() / n), "fp_other": float((loss_fp * (1 - dec_share)).sum() / n),
           "fn_blocking": float((loss_fn * blk_share).sum() / n), "fn_scoring": float((loss_fn * (1 - blk_share)).sum() / n)}
    out["total_lost"] = float(1 - f.mean())
    out["check_sum"] = float(sum(v for kk, v in out.items() if kk != "total_lost"))
    return {kk: round(v, 5) for kk, v in out.items()}

import numpy as np
from scipy.stats import permutation_test

def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom else 0.0

def association(w, A, B, kv):
    return np.mean([cosine(kv[w], kv[a]) for a in A]) - np.mean(
        [cosine(kv[w], kv[b]) for b in B]
    )

def weat_effect_size(X, Y, A, B, kv):
    X = [w for w in X if w in kv.key_to_index]
    Y = [w for w in Y if w in kv.key_to_index]
    A = [w for w in A if w in kv.key_to_index]
    B = [w for w in B if w in kv.key_to_index]

    if min(len(X), len(Y), len(A), len(B)) == 0:
        raise ValueError("One or more WEAT sets has zero vocabulary coverage.")

    sx = [association(w, A, B, kv) for w in X]
    sy = [association(w, A, B, kv) for w in Y]

    numerator = np.mean(sx) - np.mean(sy)
    pooled = np.std(sx + sy, ddof=1)
    effect = numerator / pooled if pooled else np.nan

    return {
        "X_n": len(X),
        "Y_n": len(Y),
        "A_n": len(A),
        "B_n": len(B),
        "mean_X_association": float(np.mean(sx)),
        "mean_Y_association": float(np.mean(sy)),
        "effect_size": float(effect),
    }

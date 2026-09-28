import numpy as np


def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    denom = np.linalg.norm(u) * np.linalg.norm(v)

    if denom == 0:
        return 0.0

    return float(np.dot(u, v) / denom)


def association(w, A, B, kv):
    return (
        np.mean([cosine(kv[w], kv[a]) for a in A])
        - np.mean([cosine(kv[w], kv[b]) for b in B])
    )


def weat_effect_size(X, Y, A, B, kv):

    X = [w for w in X if w in kv.key_to_index]
    Y = [w for w in Y if w in kv.key_to_index]
    A = [w for w in A if w in kv.key_to_index]
    B = [w for w in B if w in kv.key_to_index]

    if min(len(X), len(Y), len(A), len(B)) == 0:
        raise ValueError("One or more WEAT sets has zero coverage.")

    sx = np.array([association(w, A, B, kv) for w in X])
    sy = np.array([association(w, A, B, kv) for w in Y])

    numerator = np.mean(sx) - np.mean(sy)

    pooled_std = np.std(
        np.concatenate([sx, sy]),
        ddof=1
    )

    effect_size = (
        numerator / pooled_std
        if pooled_std != 0
        else np.nan
    )

    return {
        "X_n": len(X),
        "Y_n": len(Y),
        "A_n": len(A),
        "B_n": len(B),
        "mean_X_association": float(np.mean(sx)),
        "mean_Y_association": float(np.mean(sy)),
        "effect_size": float(effect_size),
    }


def weat_test(X, Y, A, B, kv, permutations=10000, seed=42):

    X = [w for w in X if w in kv.key_to_index]
    Y = [w for w in Y if w in kv.key_to_index]
    A = [w for w in A if w in kv.key_to_index]
    B = [w for w in B if w in kv.key_to_index]

    observed = weat_effect_size(X, Y, A, B, kv)

    targets = X + Y
    n_x = len(X)

    rng = np.random.default_rng(seed)

    null_distribution = []

    for _ in range(permutations):

        shuffled = rng.permutation(targets)

        perm_X = shuffled[:n_x]
        perm_Y = shuffled[n_x:]

        result = weat_effect_size(
            perm_X,
            perm_Y,
            A,
            B,
            kv
        )

        null_distribution.append(result["effect_size"])

    null_distribution = np.array(null_distribution)

    observed_effect = observed["effect_size"]

    p_value = (
        np.sum(
            np.abs(null_distribution)
            >= abs(observed_effect)
        ) + 1
    ) / (permutations + 1)

    observed["permutations"] = permutations
    observed["seed"] = seed
    observed["p_value"] = float(p_value)

    return observed
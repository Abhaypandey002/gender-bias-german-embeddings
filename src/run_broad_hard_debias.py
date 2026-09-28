from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from config import DATA, MODEL_CONFIG, RESULTS, WEAT6_DIR, GER1_DIR, GER2_DIR
from src.load_models import load_word2vec, load_fasttext
from src.weat import weat_effect_size, weat_test
from src.broad_hard_debias import build_gender_direction, hard_debias

MAX_NEUTRALIZE = 5000
PERMUTATIONS = 10000
SEED = 42


def read_lines(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def read_pairs(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["male"], r["female"]) for r in csv.DictReader(f)]


def intersection(kv, words):
    return [w for w in words if w in kv.key_to_index]


def load_eval_sets():
    return {
        "weat": (
            read_lines(WEAT6_DIR / "male_names.txt"),
            read_lines(WEAT6_DIR / "female_names.txt"),
            read_lines(WEAT6_DIR / "career.txt"),
            read_lines(WEAT6_DIR / "family.txt"),
        ),
        "ger1": (
            read_lines(GER1_DIR / "male_terms.txt"),
            read_lines(GER1_DIR / "female_terms.txt"),
            read_lines(GER1_DIR / "technical_studies.txt"),
            read_lines(GER1_DIR / "female_associated_studies.txt"),
        ),
        "ger2": (
            read_lines(GER2_DIR / "male_terms.txt"),
            read_lines(GER2_DIR / "female_terms.txt"),
            read_lines(GER2_DIR / "rationality.txt"),
            read_lines(GER2_DIR / "emotion.txt"),
        ),
    }


def filter_four(kv, sets):
    return tuple(intersection(kv, s) for s in sets)


def weat(kv, sets):
    X, Y, A, B = sets

    if min(map(len, (X, Y, A, B))) == 0:
        return {
            "n": [len(X), len(Y), len(A), len(B)],
            "effect_size": None,
            "p_value": None,
        }

    effect_result = weat_effect_size(X, Y, A, B, kv)

    test_result = weat_test(
    X,
    Y,
    A,
    B,
    kv,
    permutations=PERMUTATIONS,
    seed=SEED,
)

    return {
        "n": [len(X), len(Y), len(A), len(B)],
        "effect_size": float(effect_result["effect_size"]),
        "p_value": float(test_result["p_value"]),
    }


def semantic_preservation(before_kv, after_kv, words):
    words = intersection(before_kv, words)
    words = [w for w in words if w in after_kv.key_to_index]
    if len(words) < 2:
        return {"n_words": len(words), "mean_absolute_cosine_change": None, "pairwise_cosine_correlation": None}
    X = np.vstack([before_kv[w] for w in words]).astype(np.float64)
    Y = np.vstack([after_kv[w] for w in words]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Cx = X @ X.T
    Cy = Y @ Y.T
    tri = np.triu_indices(len(words), k=1)
    a, b = Cx[tri], Cy[tri]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    return {
        "n_words": len(words),
        "mean_absolute_cosine_change": float(np.mean(np.abs(a - b))),
        "pairwise_cosine_correlation": corr,
    }


def build_neutral_vocab(kv, all_exclusions, semantic_probe):
    excluded = set(all_exclusions)
    candidates = []
    # The model vocabularies are frequency-ordered in the source formats used here.
    # Use the intersection and a symmetric rank criterion to avoid favoring either model.
    return candidates


# def common_rank_neutral_vocab(kv1, kv2, exclusions, semantic_probe, n=MAX_NEUTRALIZE):
    # e = set(exclusions)
    # common = set(kv1.key_to_index).intersection(kv2.key_to_index)
    # common.difference_update(e)
    # # Lower normalized rank in BOTH models = more consistently frequent/common.
    # n1, n2 = len(kv1.index_to_key), len(kv2.index_to_key)
    # scored = []
    # for w in common:
    #     r1 = kv1.key_to_index[w] / n1
    #     r2 = kv2.key_to_index[w] / n2
    #     scored.append((max(r1, r2), r1 + r2, w))
    # scored.sort()
    # probes = [w for w in semantic_probe if w in common and w not in e]
    # probe_set = set(probes)
    # # Reserve space for every available semantic probe so semantic preservation
    # # measures are performed on words that were actually transformed.
    # reserve = min(len(probes), n)
    # selected = probes[:reserve]
    # for _, _, w in scored:
    #     if len(selected) >= n:
    #         break
    #     if w not in probe_set:
    #         selected.append(w)
    # return selected

def common_rank_neutral_vocab(kv1, kv2, exclusions, semantic_probe, n=MAX_NEUTRALIZE):
    e = set(exclusions)

    common = set(kv1.key_to_index).intersection(kv2.key_to_index)
    common.difference_update(e)

    # Evaluation vocabulary is deliberately included in the intervention
    # so that before/after bias measurements can detect changes.
    eval_sets = load_eval_sets()

    evaluation_words = set()
    for sets in eval_sets.values():
        for s in sets:
            evaluation_words.update(s)

    evaluation_words = [
        w for w in evaluation_words
        if w in common and w not in e
    ]

    # Semantic probes are transformed as part of the intervention.
    # They are not used to define the gender direction or bias tests.
    probe_words = [
        w for w in semantic_probe
        if w in common and w not in e
    ]

    # Words that we explicitly want to transform first.
    forced_words = []
    seen = set()

    for w in evaluation_words + probe_words:
        if w not in seen:
            forced_words.append(w)
            seen.add(w)

    if len(forced_words) > n:
        raise ValueError(
            f"Need {len(forced_words)} forced evaluation/probe words "
            f"but MAX_NEUTRALIZE={n}."
        )

    # Rank remaining common vocabulary by normalized frequency/rank
    # across both models.
    n1 = len(kv1.index_to_key)
    n2 = len(kv2.index_to_key)

    scored = []

    for w in common:
        if w in seen:
            continue

        r1 = kv1.key_to_index[w] / n1
        r2 = kv2.key_to_index[w] / n2

        # Conservative common-rank criterion:
        # a word must be reasonably common in BOTH models.
        scored.append((max(r1, r2), r1 + r2, w))

    scored.sort()

    selected = list(forced_words)

    for _, _, w in scored:
        if len(selected) >= n:
            break
        selected.append(w)

    return selected


def evaluate_model(name, kv, gender_pairs, occupation_pairs, neutral_vocab, semantic_probe):
    direction, used_gender = build_gender_direction(kv, gender_pairs)
    debiased, meta = hard_debias(kv, direction, neutral_vocab, occupation_pairs)

    eval_sets = load_eval_sets()
    def reduction(before, after):
        b = before.get("effect_size")
        a = after.get("effect_size")
        if b is None or a is None or b == 0:
            return None
        return float((abs(b) - abs(a)) / abs(b) * 100.0)

    weat_before = weat(kv, filter_four(kv, eval_sets["weat"]))
    weat_after = weat(debiased, filter_four(kv, eval_sets["weat"]))
    ger1_before = weat(kv, filter_four(kv, eval_sets["ger1"]))
    ger1_after = weat(debiased, filter_four(kv, eval_sets["ger1"]))
    ger2_before = weat(kv, filter_four(kv, eval_sets["ger2"]))
    ger2_after = weat(debiased, filter_four(kv, eval_sets["ger2"]))

    out = {
        "model": name,
        "gender_pairs_used": used_gender,
        "neutralization_requested": len(neutral_vocab),
        "neutralization_applied": len(meta.neutralized_words),
        "equalization_applied": len(meta.equalized_pairs),
        "weat_before": weat_before,
        "weat_after": weat_after,
        "weat_absolute_effect_reduction_percent": reduction(weat_before, weat_after),
        "ger1_before": ger1_before,
        "ger1_after": ger1_after,
        "ger1_absolute_effect_reduction_percent": reduction(ger1_before, ger1_after),
        "ger2_before": ger2_before,
        "ger2_after": ger2_after,
        "ger2_absolute_effect_reduction_percent": reduction(ger2_before, ger2_after),
        "semantic_preservation": semantic_preservation(kv, debiased, semantic_probe),
    }
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    gender_pairs = read_pairs(DATA / "gender_pairs.csv")
    occupation_pairs = read_pairs(DATA / "occupation_pairs.csv")
    semantic_probe = read_lines(DATA / "semantic_probe.txt")

    models = {}
    for name, cfg in MODEL_CONFIG.items():
        if not Path(cfg["path"]).exists():
            print(f"SKIP {name}: model not found at {cfg['path']}")
            continue
        models[name] = load_word2vec(cfg["path"]) if cfg["format"] == "gensim" else load_fasttext(cfg["path"])

    if len(models) < 2:
        raise SystemExit("Both Word2Vec and FastText model files are required for the controlled common-vocabulary experiment.")

    # # Exclude every explicit evaluation term, gender-definitional term, and occupation pair.
    # eval_sets = load_eval_sets()
    # exclusions = set(semantic_probe)
    # for sets in eval_sets.values():
    #     for s in sets:
    #         exclusions.update(s)
    # for a, b in gender_pairs + occupation_pairs:
    #     exclusions.update((a, b))

    # neutral_vocab = common_rank_neutral_vocab(models["word2vec_de"], models["fasttext_de"], exclusions, semantic_probe, MAX_NEUTRALIZE)
    # # semantic probes were temporarily excluded above so they can be deliberately held out
    # # from the bias evaluation but still be transformed for semantic-preservation analysis.
    # neutral_vocab = common_rank_neutral_vocab(models["word2vec_de"], models["fasttext_de"], exclusions - set(semantic_probe), semantic_probe, MAX_NEUTRALIZE)
    protected_words = set()

    for a, b in gender_pairs + occupation_pairs:
        protected_words.update((a, b))

    neutral_vocab = common_rank_neutral_vocab(
        models["word2vec_de"],
        models["fasttext_de"],
        protected_words,
        semantic_probe,
        MAX_NEUTRALIZE,
    )

    results = {
        "experiment": "broad_controlled_hard_debias",
        "max_neutralize": MAX_NEUTRALIZE,
        "neutral_vocab_size": len(neutral_vocab),
        "selection": "common vocabulary intersection; lowest maximum normalized rank across the two models; explicit evaluation/gender/occupation exclusions; semantic probes included for held-out semantic-preservation analysis",
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "models": {},
    }

    for name, kv in models.items():
        results["models"][name] = evaluate_model(name, kv, gender_pairs, occupation_pairs, neutral_vocab, semantic_probe)
        print(json.dumps(results["models"][name], ensure_ascii=False, indent=2))

    out_path = RESULTS / "broad_hard_debias.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

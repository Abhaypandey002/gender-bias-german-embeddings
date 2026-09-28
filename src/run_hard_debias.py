import json
from pathlib import Path

import pandas as pd

from config import MODEL_CONFIG, RESULTS
from src.load_models import load_word2vec, load_fasttext
from src.hard_debias import hard_debias
from src.weat import weat_test
from src.run_pca import (
    get_common_gender_pairs,
    get_common_occupation_pairs,
    run_pca_for_model,
)


def load_models():

    models = {}

    for name, cfg in MODEL_CONFIG.items():

        path = cfg["path"]

        if not path.exists():
            print(
                f"Skipping {name}: "
                f"model not found at {path}"
            )
            continue

        if cfg["format"] == "gensim":
            models[name] = load_word2vec(path)
        else:
            models[name] = load_fasttext(path)

    return models


def load_words(path):

    return [
        word.strip()
        for word in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if word.strip()
    ]


def main():

    RESULTS.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Load evaluation data
    # ---------------------------------------------------------

    gender_pairs_df = pd.read_csv(
        "data/gender_pairs.csv"
    )

    occupation_pairs_df = pd.read_csv(
        "data/occupation_pairs.csv"
    )

    # ---------------------------------------------------------
    # Load WEAT sets
    # ---------------------------------------------------------

    male_names = load_words(
        "data/weat6_ger2/male_names.txt"
    )

    female_names = load_words(
        "data/weat6_ger2/female_names.txt"
    )

    career = load_words(
        "data/weat6_ger2/career.txt"
    )

    family = load_words(
        "data/weat6_ger2/family.txt"
    )

    models = load_models()

    if len(models) < 2:
        raise RuntimeError(
            "Need both Word2Vec and FastText."
        )

    # ---------------------------------------------------------
    # CONTROLLED EVALUATION SETS
    # ---------------------------------------------------------

    common_gender_pairs = (
        get_common_gender_pairs(
            gender_pairs_df,
            models
        )
    )

    common_occupation_pairs = (
        get_common_occupation_pairs(
            occupation_pairs_df,
            models
        )
    )

    # WEAT common vocabulary
    w2v = models["word2vec_de"]
    ft = models["fasttext_de"]

    X = [
        w for w in male_names
        if w in w2v.key_to_index
        and w in ft.key_to_index
    ]

    Y = [
        w for w in female_names
        if w in w2v.key_to_index
        and w in ft.key_to_index
    ]

    A = [
        w for w in career
        if w in w2v.key_to_index
        and w in ft.key_to_index
    ]

    B = [
        w for w in family
        if w in w2v.key_to_index
        and w in ft.key_to_index
    ]

    # ---------------------------------------------------------
    # NEUTRAL WORDS
    # ---------------------------------------------------------

    neutral_words = list(
        dict.fromkeys(A + B)
    )

    print("\nHARD DEBIASING EXPERIMENT")
    print("=" * 70)

    print(
        "\nGender pairs:",
        len(common_gender_pairs)
    )

    print(
        "Occupation equality pairs:",
        len(common_occupation_pairs)
    )

    print(
        "Neutral words:",
        len(neutral_words)
    )

    print(
        "\nNeutralization targets:"
    )

    for word in neutral_words:
        print(" ", word)

    # ---------------------------------------------------------
    # RUN BOTH MODELS
    # ---------------------------------------------------------

    for name, original_kv in models.items():

        print("\n" + "=" * 70)
        print("MODEL:", name)
        print("=" * 70)

        # -----------------------------------------------------
        # BASELINE WEAT
        # -----------------------------------------------------

        baseline = weat_test(
            X,
            Y,
            A,
            B,
            original_kv,
            permutations=10000,
            seed=42
        )

        # -----------------------------------------------------
        # HARD DEBIAS
        # -----------------------------------------------------

        (
            debiased_kv,
            gender_direction,
            neutralized_count,
            equalized_count
        ) = hard_debias(
            original_kv,
            common_gender_pairs,
            neutral_words,
            common_occupation_pairs
        )

        # -----------------------------------------------------
        # DEBIASED WEAT
        # -----------------------------------------------------

        debiased = weat_test(
            X,
            Y,
            A,
            B,
            debiased_kv,
            permutations=10000,
            seed=42
        )

        # -----------------------------------------------------
        # PCA BEFORE
        # -----------------------------------------------------

        pca_before = run_pca_for_model(
            name,
            original_kv,
            common_gender_pairs,
            common_occupation_pairs
        )

        # -----------------------------------------------------
        # PCA AFTER
        # -----------------------------------------------------

        pca_after = run_pca_for_model(
            f"{name}_hard_debiased",
            debiased_kv,
            common_gender_pairs,
            common_occupation_pairs
        )

        # -----------------------------------------------------
        # CHANGE METRICS
        # -----------------------------------------------------

        effect_change = (
            debiased["effect_size"]
            - baseline["effect_size"]
        )

        absolute_effect_reduction = (
            abs(baseline["effect_size"])
            -
            abs(debiased["effect_size"])
        )

        if baseline["effect_size"] != 0:

            relative_effect_reduction = (
                absolute_effect_reduction
                /
                abs(baseline["effect_size"])
            )

        else:
            relative_effect_reduction = None

        # -----------------------------------------------------
        # SAVE RESULT
        # -----------------------------------------------------

        result = {

            "model": name,

            "method": "hard_debias",

            "evaluation": {
                "gender_pairs": common_gender_pairs,
                "occupation_pairs": (
                    common_occupation_pairs
                ),
                "weat_X": X,
                "weat_Y": Y,
                "weat_A": A,
                "weat_B": B,
                "neutralization_words": neutral_words,
            },

            "debiased_counts": {
                "neutralized": neutralized_count,
                "equalized": equalized_count,
            },

            "baseline_weat": baseline,

            "hard_debiased_weat": debiased,

            "change": {
                "effect_size_change": effect_change,
                "absolute_effect_reduction": (
                    absolute_effect_reduction
                ),
                "relative_effect_reduction": (
                    relative_effect_reduction
                ),
            },

            "baseline_pca": pca_before,

            "hard_debiased_pca": pca_after,
        }

        output = (
            RESULTS /
            f"hard_debias_{name}.json"
        )

        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        # -----------------------------------------------------
        # PRINT SUMMARY
        # -----------------------------------------------------

        print("\nWEAT BEFORE")
        print(
            f"Effect size: "
            f"{baseline['effect_size']:.6f}"
        )

        print(
            f"p-value: "
            f"{baseline['p_value']:.6f}"
        )

        print("\nWEAT AFTER HARD DEBIAS")
        print(
            f"Effect size: "
            f"{debiased['effect_size']:.6f}"
        )

        print(
            f"p-value: "
            f"{debiased['p_value']:.6f}"
        )

        print("\nCHANGE")

        print(
            f"Absolute effect reduction: "
            f"{absolute_effect_reduction:.6f}"
        )

        if relative_effect_reduction is not None:

            print(
                f"Relative effect reduction: "
                f"{relative_effect_reduction * 100:.2f}%"
            )

        print("\nPCA BEFORE")
        print(
            f"Explained variance: "
            f"{pca_before['explained_variance_ratio']:.6f}"
        )

        print("\nPCA AFTER")
        print(
            f"Explained variance: "
            f"{pca_after['explained_variance_ratio']:.6f}"
        )

        print(
            "\nNeutralized:",
            neutralized_count
        )

        print(
            "Equalized:",
            equalized_count
        )


if __name__ == "__main__":
    main()
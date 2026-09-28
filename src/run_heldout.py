import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_CONFIG, RESULTS
from src.load_models import load_word2vec, load_fasttext
from src.hard_debias import hard_debias
from src.weat import weat_test
from src.run_pca import get_common_gender_pairs


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


def common_words(words, models):

    return [
        word
        for word in words
        if all(
            word in kv.key_to_index
            for kv in models.values()
        )
    ]


def common_weat_sets(
    X,
    Y,
    A,
    B,
    models
):

    return (
        common_words(X, models),
        common_words(Y, models),
        common_words(A, models),
        common_words(B, models),
    )


def pairwise_cosine_matrix(words, kv):

    vectors = np.asarray(
        [kv[word] for word in words],
        dtype=float
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    normalized = vectors / np.maximum(
        norms,
        1e-12
    )

    return normalized @ normalized.T


def semantic_preservation(
    words,
    original_kv,
    debiased_kv
):

    words = [
        word
        for word in words
        if (
            word in original_kv.key_to_index
            and word in debiased_kv.key_to_index
        )
    ]

    if len(words) < 2:
        return {
            "words_used": len(words),
            "mean_absolute_change": None,
            "correlation": None,
        }

    before = pairwise_cosine_matrix(
        words,
        original_kv
    )

    after = pairwise_cosine_matrix(
        words,
        debiased_kv
    )

    # Only upper triangle; don't count diagonal.
    mask = np.triu(
        np.ones(before.shape, dtype=bool),
        k=1
    )

    before_values = before[mask]
    after_values = after[mask]

    mean_absolute_change = float(
        np.mean(
            np.abs(
                before_values -
                after_values
            )
        )
    )

    if (
        np.std(before_values) == 0
        or np.std(after_values) == 0
    ):
        correlation = None
    else:
        correlation = float(
            np.corrcoef(
                before_values,
                after_values
            )[0, 1]
        )

    return {
        "words_used": len(words),
        "mean_absolute_change": (
            mean_absolute_change
        ),
        "correlation": correlation,
    }


def main():

    RESULTS.mkdir(
        exist_ok=True
    )

    models = load_models()

    if len(models) < 2:
        raise RuntimeError(
            "Need both Word2Vec and FastText."
        )

    # ---------------------------------------------------------
    # Gender pairs used to construct debias direction
    # ---------------------------------------------------------

    gender_pairs_df = pd.read_csv(
        "data/gender_pairs.csv"
    )

    common_gender_pairs = (
        get_common_gender_pairs(
            gender_pairs_df,
            models
        )
    )

    # ---------------------------------------------------------
    # Occupation pairs used for equalization
    # ---------------------------------------------------------

    occupation_df = pd.read_csv(
        "data/occupation_pairs.csv"
    )

    common_occupation_pairs = []

    for _, row in occupation_df.iterrows():

        male = row["male"]
        female = row["female"]

        if all(
            male in kv.key_to_index
            and female in kv.key_to_index
            for kv in models.values()
        ):
            common_occupation_pairs.append(
                (male, female)
            )

    # ---------------------------------------------------------
    # Same neutralization list used in Hard Debias
    # ---------------------------------------------------------

    career = load_words(
        "data/weat6_ger2/career.txt"
    )

    family = load_words(
        "data/weat6_ger2/family.txt"
    )

    neutral_words = list(
        dict.fromkeys(
            common_words(
                career + family,
                models
            )
        )
    )

    # ---------------------------------------------------------
    # GER1
    # ---------------------------------------------------------

    ger1_male = load_words(
        "data/ger1/male_terms.txt"
    )

    ger1_female = load_words(
        "data/ger1/female_terms.txt"
    )

    ger1_technical = load_words(
        "data/ger1/technical_studies.txt"
    )

    ger1_female_studies = load_words(
        "data/ger1/female_associated_studies.txt"
    )

    (
        ger1_male,
        ger1_female,
        ger1_technical,
        ger1_female_studies,
    ) = common_weat_sets(
        ger1_male,
        ger1_female,
        ger1_technical,
        ger1_female_studies,
        models
    )

    # ---------------------------------------------------------
    # GER2
    # ---------------------------------------------------------

    ger2_male = load_words(
        "data/ger2/male_terms.txt"
    )

    ger2_female = load_words(
        "data/ger2/female_terms.txt"
    )

    ger2_rationality = load_words(
        "data/ger2/rationality.txt"
    )

    ger2_emotion = load_words(
        "data/ger2/emotion.txt"
    )

    (
        ger2_male,
        ger2_female,
        ger2_rationality,
        ger2_emotion,
    ) = common_weat_sets(
        ger2_male,
        ger2_female,
        ger2_rationality,
        ger2_emotion,
        models
    )

    print("\nHELD-OUT GER1 + GER2 EVALUATION")
    print("=" * 70)

    print("\nGER1 common sets:")
    print("Male:", ger1_male)
    print("Female:", ger1_female)
    print("Technical:", ger1_technical)
    print("Female-associated studies:", ger1_female_studies)

    print("\nGER2 common sets:")
    print("Male:", ger2_male)
    print("Female:", ger2_female)
    print("Rationality:", ger2_rationality)
    print("Emotion:", ger2_emotion)

    # ---------------------------------------------------------
    # Run each model
    # ---------------------------------------------------------

    for name, original_kv in models.items():

        print("\n" + "=" * 70)
        print("MODEL:", name)
        print("=" * 70)

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
        # GER1
        # -----------------------------------------------------

        ger1_before = weat_test(
            ger1_male,
            ger1_female,
            ger1_technical,
            ger1_female_studies,
            original_kv,
            permutations=10000,
            seed=42
        )

        ger1_after = weat_test(
            ger1_male,
            ger1_female,
            ger1_technical,
            ger1_female_studies,
            debiased_kv,
            permutations=10000,
            seed=42
        )

        # -----------------------------------------------------
        # GER2
        # -----------------------------------------------------

        ger2_before = weat_test(
            ger2_male,
            ger2_female,
            ger2_rationality,
            ger2_emotion,
            original_kv,
            permutations=10000,
            seed=42
        )

        ger2_after = weat_test(
            ger2_male,
            ger2_female,
            ger2_rationality,
            ger2_emotion,
            debiased_kv,
            permutations=10000,
            seed=42
        )

        # -----------------------------------------------------
        # Held-out semantic words
        #
        # These are NOT neutralized or equalized.
        # -----------------------------------------------------

        semantic_words = list(
            dict.fromkeys(
                ger1_technical
                + ger1_female_studies
                + ger2_rationality
                + ger2_emotion
            )
        )

        semantic_result = semantic_preservation(
            semantic_words,
            original_kv,
            debiased_kv
        )

        # -----------------------------------------------------
        # Relative reductions
        # -----------------------------------------------------

        ger1_reduction = (
            abs(ger1_before["effect_size"])
            -
            abs(ger1_after["effect_size"])
        )

        ger2_reduction = (
            abs(ger2_before["effect_size"])
            -
            abs(ger2_after["effect_size"])
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        result = {

            "model": name,

            "method": "hard_debias",

            "held_out_evaluation": True,

            "ger1": {
                "before": ger1_before,
                "after": ger1_after,
                "absolute_effect_reduction": (
                    float(ger1_reduction)
                ),
            },

            "ger2": {
                "before": ger2_before,
                "after": ger2_after,
                "absolute_effect_reduction": (
                    float(ger2_reduction)
                ),
            },

            "semantic_preservation": semantic_result,

            "debiased_counts": {
                "neutralized": neutralized_count,
                "equalized": equalized_count,
            },
        }

        output = (
            RESULTS /
            f"heldout_hard_debias_{name}.json"
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
        # Print
        # -----------------------------------------------------

        print("\nGER1 BEFORE")
        print(
            f"Effect size: "
            f"{ger1_before['effect_size']:.6f}"
        )
        print(
            f"p-value: "
            f"{ger1_before['p_value']:.6f}"
        )

        print("\nGER1 AFTER")
        print(
            f"Effect size: "
            f"{ger1_after['effect_size']:.6f}"
        )
        print(
            f"p-value: "
            f"{ger1_after['p_value']:.6f}"
        )

        print(
            f"GER1 absolute reduction: "
            f"{ger1_reduction:.6f}"
        )

        print("\nGER2 BEFORE")
        print(
            f"Effect size: "
            f"{ger2_before['effect_size']:.6f}"
        )
        print(
            f"p-value: "
            f"{ger2_before['p_value']:.6f}"
        )

        print("\nGER2 AFTER")
        print(
            f"Effect size: "
            f"{ger2_after['effect_size']:.6f}"
        )
        print(
            f"p-value: "
            f"{ger2_after['p_value']:.6f}"
        )

        print(
            f"GER2 absolute reduction: "
            f"{ger2_reduction:.6f}"
        )

        print("\nSEMANTIC PRESERVATION")

        print(
            "Words used:",
            semantic_result["words_used"]
        )

        print(
            "Mean absolute cosine change:",
            semantic_result[
                "mean_absolute_change"
            ]
        )

        print(
            "Before/after cosine correlation:",
            semantic_result[
                "correlation"
            ]
        )


if __name__ == "__main__":
    main()
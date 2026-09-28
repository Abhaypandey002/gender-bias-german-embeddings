import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import GENDER_PAIRS, RESULTS, MODEL_CONFIG
from src.load_models import load_word2vec, load_fasttext


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


def normalize(v):
    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def cosine_to_direction(v, direction):
    v = normalize(v)
    direction = normalize(direction)

    return float(np.dot(v, direction))


def get_common_gender_pairs(gender_pairs, models):
    """
    Keep only gender pairs where BOTH models
    contain BOTH words.
    """

    common_pairs = []

    for _, row in gender_pairs.iterrows():

        male = row["male"]
        female = row["female"]

        if all(
            male in kv.key_to_index
            and female in kv.key_to_index
            for kv in models.values()
        ):
            common_pairs.append(
                (male, female)
            )

    return common_pairs


def get_common_occupation_pairs(occupation_pairs, models):
    """
    Keep only occupation pairs where BOTH models
    contain BOTH words.
    """

    common_pairs = []

    for _, row in occupation_pairs.iterrows():

        male = row["male"]
        female = row["female"]

        if all(
            male in kv.key_to_index
            and female in kv.key_to_index
            for kv in models.values()
        ):
            common_pairs.append(
                (male, female)
            )

    return common_pairs


def run_pca_for_model(
    name,
    kv,
    gender_pairs,
    occupation_pairs
):

    # ---------------------------------------------------------
    # 1. Gender difference vectors
    # ---------------------------------------------------------

    difference_vectors = []

    for male, female in gender_pairs:

        vector = (
            kv[male] -
            kv[female]
        )

        difference_vectors.append(vector)

    difference_vectors = np.asarray(
        difference_vectors
    )

    # ---------------------------------------------------------
    # 2. PCA
    # ---------------------------------------------------------

    pca = PCA(n_components=1)

    pca.fit(difference_vectors)

    gender_direction = pca.components_[0]

    # Orient PCA direction consistently:
    # Mann should have a higher score than Frau.

    if (
        "Mann" in kv.key_to_index
        and "Frau" in kv.key_to_index
    ):

        mann_score = cosine_to_direction(
            kv["Mann"],
            gender_direction
        )

        frau_score = cosine_to_direction(
            kv["Frau"],
            gender_direction
        )

        if mann_score < frau_score:
            gender_direction *= -1

    gender_direction = normalize(
        gender_direction
    )

    explained_variance = float(
        pca.explained_variance_ratio_[0]
    )

    # ---------------------------------------------------------
    # 3. Occupation projections
    # ---------------------------------------------------------

    occupation_results = []

    for male, female in occupation_pairs:

        male_score = cosine_to_direction(
            kv[male],
            gender_direction
        )

        female_score = cosine_to_direction(
            kv[female],
            gender_direction
        )

        occupation_results.append({
            "male": male,
            "female": female,
            "male_gender_direction_score": male_score,
            "female_gender_direction_score": female_score,
            "pair_difference": (
                male_score - female_score
            ),
        })

    # ---------------------------------------------------------
    # 4. Save JSON
    # ---------------------------------------------------------

    result = {
        "model": name,
        "gender_pairs_used": len(gender_pairs),
        "gender_pairs": [
            {
                "male": male,
                "female": female
            }
            for male, female in gender_pairs
        ],
        "pca_components": 1,
        "explained_variance_ratio": explained_variance,
        "occupation_pairs_used": len(
            occupation_pairs
        ),
        "occupation_pairs": [
            {
                "male": male,
                "female": female
            }
            for male, female in occupation_pairs
        ],
        "occupation_results": occupation_results,
    }

    output = (
        RESULTS /
        f"gender_direction_pca_common_{name}.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # 5. Save occupation CSV
    # ---------------------------------------------------------

    csv_output = (
        RESULTS /
        f"gender_direction_occupations_common_{name}.csv"
    )

    pd.DataFrame(
        occupation_results
    ).to_csv(
        csv_output,
        index=False,
        encoding="utf-8"
    )

    return result


def main():

    RESULTS.mkdir(
        exist_ok=True
    )

    gender_pairs = pd.read_csv(
        GENDER_PAIRS
    )

    occupation_pairs = pd.read_csv(
        Path("data/occupation_pairs.csv")
    )

    models = load_models()

    if len(models) < 2:
        raise RuntimeError(
            "Need both Word2Vec and FastText "
            "for a controlled comparison."
        )

    # ---------------------------------------------------------
    # COMMON VOCABULARY
    # ---------------------------------------------------------

    common_gender_pairs = get_common_gender_pairs(
        gender_pairs,
        models
    )

    common_occupation_pairs = (
        get_common_occupation_pairs(
            occupation_pairs,
            models
        )
    )

    print("\nCONTROLLED GENDER DIRECTION + PCA")
    print("=" * 65)

    print(
        "\nCommon gender pairs:",
        len(common_gender_pairs),
        "/",
        len(gender_pairs)
    )

    for male, female in common_gender_pairs:
        print(
            f"  {male} <-> {female}"
        )

    print(
        "\nCommon occupation pairs:",
        len(common_occupation_pairs),
        "/",
        len(occupation_pairs)
    )

    for male, female in common_occupation_pairs:
        print(
            f"  {male} <-> {female}"
        )

    # ---------------------------------------------------------
    # RUN SAME EXPERIMENT ON BOTH MODELS
    # ---------------------------------------------------------

    for name, kv in models.items():

        result = run_pca_for_model(
            name,
            kv,
            common_gender_pairs,
            common_occupation_pairs
        )

        print(
            f"\nMODEL: {name}"
        )

        print(
            "Gender pairs used:",
            result["gender_pairs_used"]
        )

        print(
            "PCA explained variance:",
            f"{result['explained_variance_ratio']:.4f}"
        )

        print(
            "Occupation pairs used:",
            result["occupation_pairs_used"]
        )

        print("\nOccupation results:")

        for row in result[
            "occupation_results"
        ]:

            print(
                f"{row['male']:15s} "
                f"{row['male_gender_direction_score']:+.4f} | "
                f"{row['female']:15s} "
                f"{row['female_gender_direction_score']:+.4f} | "
                f"difference="
                f"{row['pair_difference']:+.4f}"
            )


if __name__ == "__main__":
    main()
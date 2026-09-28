import copy
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def build_gender_direction(kv, gender_pairs):
    """
    Build a gender direction using PCA over
    male-female difference vectors.
    """

    vectors = []

    for male, female in gender_pairs:

        if (
            male not in kv.key_to_index
            or female not in kv.key_to_index
        ):
            continue

        diff = kv[male] - kv[female]

        vectors.append(diff)

    if len(vectors) < 2:
        raise ValueError(
            "Not enough gender pairs available."
        )

    vectors = np.asarray(vectors)

    # Center difference vectors
    centered = vectors - vectors.mean(axis=0)

    # SVD gives first principal direction
    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False
    )

    direction = normalize(vh[0])

    # Consistent orientation
    if (
        "Mann" in kv.key_to_index
        and "Frau" in kv.key_to_index
    ):

        mann_score = np.dot(
            normalize(kv["Mann"]),
            direction
        )

        frau_score = np.dot(
            normalize(kv["Frau"]),
            direction
        )

        if mann_score < frau_score:
            direction *= -1

    return normalize(direction)


def project_on_direction(v, direction):
    return np.dot(v, direction) * direction


def neutralize_vector(v, direction):
    """
    Remove the component of v along the gender direction.
    """

    projection = project_on_direction(
        v,
        direction
    )

    neutralized = v - projection

    return normalize(neutralized)


def equalize_pair(
    male_vector,
    female_vector,
    direction
):
    """
    Equalize a gendered pair around the
    gender-neutral component.
    """

    male_vector = normalize(male_vector)
    female_vector = normalize(female_vector)

    # Mean vector
    mean = (
        male_vector +
        female_vector
    ) / 2.0

    # Remove gender component from mean
    mean_neutral = (
        mean -
        project_on_direction(
            mean,
            direction
        )
    )

    mean_norm = np.linalg.norm(
        mean_neutral
    )

    if mean_norm == 0:
        mean_neutral = np.zeros_like(
            mean_neutral
        )

    else:
        mean_neutral = (
            mean_neutral /
            mean_norm
        )

    # Direction separating the pair
    pair_difference = (
        male_vector -
        female_vector
    )

    gender_component = project_on_direction(
        pair_difference,
        direction
    )

    gender_norm = np.linalg.norm(
        gender_component
    )

    if gender_norm == 0:
        return (
            normalize(mean_neutral),
            normalize(mean_neutral)
        )

    gender_component = (
        gender_component /
        gender_norm
    )

    # Equalized vectors
    male_new = (
        mean_neutral +
        gender_component
    )

    female_new = (
        mean_neutral -
        gender_component
    )

    return (
        normalize(male_new),
        normalize(female_new)
    )


def hard_debias(
    kv,
    gender_pairs,
    neutral_words,
    equality_pairs
):
    """
    Controlled Hard Debias.

    1. Identify gender direction.
    2. Neutralize specified neutral words.
    3. Equalize specified gendered pairs.
    """

    # Work on a copy so original embeddings
    # remain untouched.
    debiased = copy.deepcopy(kv)

    gender_direction = build_gender_direction(
        debiased,
        gender_pairs
    )

    # ---------------------------------------------------------
    # Neutralization
    # ---------------------------------------------------------

    neutralized_count = 0

    for word in neutral_words:

        if word not in debiased.key_to_index:
            continue

        idx = debiased.key_to_index[word]

        old_vector = debiased.vectors[idx]

        new_vector = neutralize_vector(
            old_vector,
            gender_direction
        )

        debiased.vectors[idx] = new_vector

        neutralized_count += 1

    # ---------------------------------------------------------
    # Equalization
    # ---------------------------------------------------------

    equalized_count = 0

    for male, female in equality_pairs:

        if (
            male not in debiased.key_to_index
            or female not in debiased.key_to_index
        ):
            continue

        male_idx = debiased.key_to_index[male]
        female_idx = debiased.key_to_index[female]

        male_vector = debiased.vectors[
            male_idx
        ]

        female_vector = debiased.vectors[
            female_idx
        ]

        new_male, new_female = equalize_pair(
            male_vector,
            female_vector,
            gender_direction
        )

        debiased.vectors[male_idx] = (
            new_male
        )

        debiased.vectors[female_idx] = (
            new_female
        )

        equalized_count += 1

    return (
        debiased,
        gender_direction,
        neutralized_count,
        equalized_count
    )
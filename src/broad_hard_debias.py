from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v.copy()
    return v / n


def build_gender_direction(kv, pairs: Iterable[Tuple[str, str]]):
    diffs = []
    used = []
    for male, female in pairs:
        if male in kv.key_to_index and female in kv.key_to_index:
            diffs.append(kv[male] - kv[female])
            used.append((male, female))
    if not diffs:
        raise ValueError("No gender pairs are available in the model vocabulary.")

    X = np.vstack(diffs).astype(np.float32)
    # PCA/SVD on definitional gender-pair differences.
    _, _, vh = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
    direction = unit(vh[0].astype(np.float32))
    return direction, used


@dataclass
class DebiasResult:
    direction: np.ndarray
    neutralized_words: List[str]
    equalized_pairs: List[Tuple[str, str]]


class DebiasedVectors:
    """Lightweight view over original vectors with sparse transformed vectors.

    This avoids copying a 2M x 300 FastText matrix just to transform a few thousand
    words. Untouched words are read directly from the original model.
    """

    def __init__(self, kv, replacements: Dict[str, np.ndarray]):
        self._kv = kv
        self._replacements = replacements
        self.key_to_index = kv.key_to_index
        self.index_to_key = kv.index_to_key

    def __contains__(self, key):
        return key in self.key_to_index

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._replacements.get(key, self._kv[key])
        return self._kv[key]

    def get_vector(self, key, *args, **kwargs):
        return self[key]


def hard_debias(kv, gender_direction, neutralize_words, equalize_pairs):
    g = unit(gender_direction.astype(np.float32))
    replacements: Dict[str, np.ndarray] = {}
    neutralized = []
    equalized = []

    # Broad neutralization: remove the gender-direction component from neutral words.
    for word in neutralize_words:
        if word not in kv.key_to_index:
            continue
        v = np.asarray(kv[word], dtype=np.float32)
        v_neutral = v - np.dot(v, g) * g
        replacements[word] = unit(v_neutral).astype(np.float32)
        neutralized.append(word)

    # Equalize gendered occupation pairs around their shared neutral component.
    for male, female in equalize_pairs:
        if male not in kv.key_to_index or female not in kv.key_to_index:
            continue
        vm = np.asarray(kv[male], dtype=np.float32)
        vf = np.asarray(kv[female], dtype=np.float32)
        mean = (vm + vf) / 2.0
        mean_perp = mean - np.dot(mean, g) * g
        norm_perp = np.linalg.norm(mean_perp)
        alpha = np.sqrt(max(0.0, 1.0 - float(norm_perp) ** 2))
        sm = 1.0 if np.dot(vm, g) >= 0 else -1.0
        sf = 1.0 if np.dot(vf, g) >= 0 else -1.0
        # If both signs are identical, force opposite signs using the original ordering.
        if sm == sf:
            sf = -sm
        replacements[male] = unit(mean_perp + sm * alpha * g).astype(np.float32)
        replacements[female] = unit(mean_perp + sf * alpha * g).astype(np.float32)
        equalized.append((male, female))

    return DebiasedVectors(kv, replacements), DebiasResult(
        direction=g,
        neutralized_words=neutralized,
        equalized_pairs=equalized,
    )

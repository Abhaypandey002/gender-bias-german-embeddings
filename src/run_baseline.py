import json

# from config import WEAT6_DIR, MODEL_CONFIG, RESULTS
# from load_models import load_word2vec, load_fasttext
# from weat import weat_test

from config import WEAT6_DIR, MODEL_CONFIG, RESULTS
from src.load_models import load_word2vec, load_fasttext
# from src.weat import weat_effect_size
from src.weat import weat_effect_size, weat_test

def words(filename):

    path = WEAT6_DIR / filename

    return [
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def shared_words(words_a, words_b, kv_a, kv_b):

    return [
        w for w in words_a
        if w in kv_a.key_to_index
        and w in kv_b.key_to_index
        and w in words_b
    ]


def main():

    RESULTS.mkdir(exist_ok=True)

    male_names = words("male_names.txt")
    female_names = words("female_names.txt")
    career = words("career.txt")
    family = words("family.txt")

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

    if len(models) < 2:
        raise RuntimeError(
            "Need both Word2Vec and FastText "
            "for a controlled comparison."
        )

    w2v = models["word2vec_de"]
    ft = models["fasttext_de"]

    # Common vocabulary across BOTH models
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

    print("\nCOMMON WEAT6-GER2 SETS")
    print("Male names:", X)
    print("Female names:", Y)
    print("Career:", A)
    print("Family:", B)

    for name, kv in models.items():

        result = weat_test(
            X,
            Y,
            A,
            B,
            kv,
            permutations=10000,
            seed=42
        )

        result["model"] = name
        result["test"] = "WEAT6-GER2"

        output = (
            RESULTS /
            f"baseline_weat6_common_{name}.json"
        )

        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print("\n", name)
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
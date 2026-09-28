from pathlib import Path
import json
from config import WEAT6_DIR, MODEL_CONFIG, RESULTS
# from load_models import load_word2vec, load_fasttext
from src.load_models import load_word2vec, load_fasttext
# from weat import weat_effect_size
from src.weat import weat_effect_size

def words(name):
    return [
        x.strip() for x in (WEAT6_DIR / name).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

def main():
    RESULTS.mkdir(exist_ok=True)

    X = words("male_names.txt")
    Y = words("female_names.txt")
    A = words("career.txt")
    B = words("family.txt")

    for name, cfg in MODEL_CONFIG.items():
        path = cfg["path"]
        if not path.exists():
            print(f"Skipping {name}: model not found at {path}")
            continue

        kv = load_word2vec(path) if cfg["format"] == "gensim" else load_fasttext(path)
        result = weat_effect_size(X, Y, A, B, kv)
        result["model"] = name

        out = RESULTS / f"baseline_weat6_{name}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

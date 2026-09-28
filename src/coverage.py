import csv
from pathlib import Path
from config import DATA, MODEL_CONFIG
# from load_models import load_word2vec, load_fasttext
from src.load_models import load_word2vec, load_fasttext

def read_words(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def all_words():
    words = set()
    for p in DATA.rglob("*.txt"):
        words.update(read_words(p))
    p = DATA / "gender_pairs.csv"
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words.update([row["male"], row["female"]])
    with (DATA / "occupation_pairs.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words.update([row["male"], row["female"]])
    return sorted(words)

def main():
    words = all_words()
    print(f"Total unique evaluation words: {len(words)}")

    for name, cfg in MODEL_CONFIG.items():
        path = cfg["path"]
        if not path.exists():
            print(f"\n{name}: MODEL FILE NOT FOUND -> {path}")
            continue

        if cfg["format"] == "gensim":
            kv = load_word2vec(path)
        else:
            kv = load_fasttext(path)

        present = [w for w in words if w in kv.key_to_index]
        missing = [w for w in words if w not in kv.key_to_index]

        print(f"\n{name}")
        print(f"Vocabulary size: {len(kv.key_to_index):,}")
        print(f"Evaluation coverage: {len(present)}/{len(words)} ({len(present)/len(words):.1%})")
        print("Missing:", ", ".join(missing) if missing else "None")

if __name__ == "__main__":
    main()

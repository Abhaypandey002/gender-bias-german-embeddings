from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODEL_CONFIG = {
    "word2vec_de": {
        "path": MODELS / "german.model",
        "format": "gensim",
    },
    "fasttext_de": {
        "path": MODELS / "cc.de.300.bin",
        "format": "fasttext",
    },
}

GENDER_PAIRS = DATA / "gender_pairs.csv"
WEAT6_DIR = DATA / "weat6_ger2"
GER1_DIR = DATA / "ger1"
GER2_DIR = DATA / "ger2"

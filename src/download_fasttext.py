import requests
from pathlib import Path
from config import MODELS

URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
OUT = MODELS / "cc.de.300.bin.gz"

def main():
    MODELS.mkdir(exist_ok=True)
    print("Downloading German fastText model (~GB-scale file).")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with OUT.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved to {OUT}")
    print("Unzip it with a tool such as 7-Zip to obtain cc.de.300.bin")

if __name__ == "__main__":
    main()

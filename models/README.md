@"
# Pretrained Models

The large pretrained embedding models are intentionally not stored in this Git repository because of their file size.

## German Word2Vec

Expected file:

`german.model`

Place it at:

`models/german.model`

Download it from the GermanWordEmbeddings project:

https://github.com/devmount/GermanWordEmbeddings

## German FastText

Expected file:

`cc.de.300.bin`

Place it at:

`models/cc.de.300.bin`

The model is the official German Common Crawl + Wikipedia FastText model.

Download the German crawl vectors from:

https://fasttext.cc/docs/en/crawl-vectors.html

After downloading `cc.de.300.bin.gz`, extract it to obtain:

`cc.de.300.bin`

## Expected directory

```text
models/
├── german.model
└── cc.de.300.bin
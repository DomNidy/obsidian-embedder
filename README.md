# obsidian-embedder

Generate document embeddings for your Obsidian vault and organize them into groups (bins) based on semantic similarity. This utility is designed to help manage large Obsidian vaults with many documents, making it easier to identify redundant or poorly categorized documents.

The similarity search/document binning is powered by a Sentence-BERT model.

_(This README is a W.I.P)_

## Step 1: Pre-processing your Obsidian vault

To prepare your documents to be embedded, you can run the `preprocess.py` script and point it at the root directory of your Obsidian vault. The output of this script will be a single text file where each line contains the contents of a single document chunk.

_**Note:** Currently, creating multiple chunks from a single document is not supported. Each chunk in the output text file will be the contents of an entire document._

**Example:**

```sh
python preprocess.py C:/vault -mlr 0.30 -mdl 150 -bw tags review srdue srease excalidraw --include-document-title -o preprocessed_vault.txt
```

## Step 2: Embed and bin documents

_TODO_

**Example:**

```sh
python bin_documents.py preprocessed_vault.txt bins.txt
```

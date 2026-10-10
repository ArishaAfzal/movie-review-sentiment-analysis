# Movie Review Sentiment Analysis

**Movie-review sentiment classification with a custom PyTorch transformer.**

This project explores the IMDb review dataset, builds a transformer classifier from scratch, and evaluates positive/negative sentiment predictions on held-out reviews. The saved notebook run achieved **77.58% test accuracy**.

[Explore the notebook](Movie_Review_Sentiment_Analysis.ipynb) · [Portfolio](https://arishaafzal.github.io/#work)

## Results

| Metric | Recorded result |
| --- | ---: |
| Final validation accuracy | 80.76% |
| Test accuracy | 77.58% |
| Epochs | 5 |
| Training reviews | 22,500 |
| Validation reviews | 2,500 |
| Test reviews | 25,000 |

These metrics come from the saved outputs in the original notebook. The notebook was cleaned for publication without rerunning training. A trained model checkpoint is not included. Fresh-run results can vary because initialization and training shuffles are not seeded.

## Approach

- Explore class balance, review lengths, duplicates, missing values, and sample reviews with pandas and Matplotlib.
- Use the `bert-base-uncased` subword tokenizer with a maximum sequence length of 128 tokens.
- Train a custom transformer with 384-dimensional embeddings, six blocks, eight attention heads, and dropout of 0.2.
- Apply padding-aware mean pooling and a two-class linear head.
- Optimize with AdamW, cross-entropy loss, learning rate `2e-4`, and weight decay `0.01`.

Only the tokenizer is pretrained; this project does **not** fine-tune pretrained BERT model weights.

## Run locally

Use Python 3.10 or newer. A CUDA GPU is recommended for training; CPU execution is supported but slower.

```bash
git clone https://github.com/ArishaAfzal/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Download `aclImdb_v1.tar.gz` from the [Stanford Large Movie Review Dataset page](https://ai.stanford.edu/~amaas/data/sentiment/) and place it in the repository root. The extraction cell uses `tar`; on systems without it, extract the archive manually and skip that cell. The expected directories are:

```text
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

Launch Jupyter and run the notebook cells in order:

```bash
jupyter lab Movie_Review_Sentiment_Analysis.ipynb
```

Internet access is required on the first run to download the tokenizer. Dependencies are listed as compatible ranges rather than a lockfile of the original training environment.

## Limitations

Reviews are truncated to 128 tokens, which can omit sentiment cues. This is a learning project using causal attention, not a production sentiment service. The reported result represents one training run.

## Project context

Completed by Arisha Afzal as part of the **Udacity Future AWS AI Programmer Nanodegree**, adapted from the supplied SentimentScope project starter notebook. The dataset is the Stanford Large Movie Review Dataset; review its source page for dataset usage terms.

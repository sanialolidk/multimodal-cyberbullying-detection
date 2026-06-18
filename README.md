# Multimodal Cyberbullying Detection

This project implements a lightweight machine learning pipeline for cyberbullying detection using both text content and user interaction context.

## What It Builds

- NLP preprocessing: lowercasing, punctuation cleanup, tokenization, stop-word removal, and text normalization.
- Text features: TF-IDF vectors plus small lexicon-style embedding features for abusive and positive language density.
- Network features: author out-degree, author in-degree, target in-degree, repeated interaction frequency, and neighbor counts.
- Classifier: binary logistic regression implemented with NumPy so the demo runs without scikit-learn.
- Evaluation: accuracy, precision, recall, F1, confusion counts, feature weights, and example predictions.

## Files

- `sample_cyberbullying_data.csv`: small labeled demo dataset.
- `multimodal_cyberbullying_pipeline.py`: end-to-end training and evaluation script.
- `requirements.txt`: minimal Python dependencies.
- `project_summary.md`: resume/project explanation.

## Run

```bash
python3 multimodal_cyberbullying_pipeline.py --data sample_cyberbullying_data.csv
```

If you are using the bundled Codex runtime from this workspace:

```bash
/Users/saniathankan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 multimodal_cyberbullying_pipeline.py --data sample_cyberbullying_data.csv
```

## Expected Output

The script prints model metrics, the strongest cyberbullying signals, and predictions for the held-out sample rows.

## Notes

The included dataset is intentionally small and synthetic for demonstration. For a production-grade model, replace it with a larger labeled dataset, add cross-validation, calibrate thresholds, audit bias across user groups, and keep privacy safeguards around interaction data.

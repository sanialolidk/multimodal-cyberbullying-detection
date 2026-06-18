# Project Summary: Multimodal Cyberbullying Detection

Built an end-to-end ML pipeline that combines NLP-derived text features with user interaction network features to classify cyberbullying content.

## Key Work

- Preprocessed social media text using normalization, tokenization, stop-word removal, and lexical cleanup.
- Extracted TF-IDF features to capture discriminative words and phrases associated with harmful language.
- Added lightweight word embedding-style signals using abusive and positive language density, average token length, and message length.
- Engineered interaction-network features from author-target relationships, including out-degree, in-degree, repeated pair frequency, and neighborhood size.
- Combined text and graph features into one model-ready feature matrix.
- Trained a binary logistic regression classifier and evaluated accuracy, precision, recall, F1, and confusion counts.
- Reported feature weights to make the model behavior easier to inspect.

## Resume Version

Multimodal Cyberbullying Detection (NLP + Network Features)

- Built an ML pipeline combining text-based features including TF-IDF and embedding-style lexical signals with user interaction network features.
- Applied NLP preprocessing such as tokenization, stop-word removal, and text normalization using Python.
- Engineered linguistic and interaction-context features to improve cyberbullying classification performance and model interpretability.

## Tech Stack

Python, pandas, NumPy, NLP preprocessing, TF-IDF, graph feature engineering, logistic regression.

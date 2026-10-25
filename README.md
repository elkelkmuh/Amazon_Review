ModernBERT-Based Sentiment Classification
Amazon_Sentiment_Main.ipynb provides the preprocessing, training, and evaluation workflow for Dataset 1: Amazon Customer Reviews. It compares TF-IDF + SVM, TF-IDF + logistic regression, BERT-base, ModernBERT, ModernBERT + TAPT, ModernBERT + AdaptFT, and ModernBERT + Fusion.
Usage
1. Open the notebook in Google Colab or Jupyter with a CUDA-enabled GPU.
2. Replace the two path placeholders: CSV_FILE with the input CSV path and OUT_DIR with an output directory.
3. Run the code cell. Dependencies are installed automatically; internet access is required to download pretrained models. For Google Drive inputs, mount Drive separately before running.
The CSV must contain Review_text and Rating. Ratings of 1–2 are negative, ratings of 4–5 are positive, and three-star reviews are excluded.
Workflow
The notebook removes missing or invalid entries, reviews shorter than five words, and exact duplicate review texts. It uses a stratified 70/15/15 split before capping training, validation, and test subsets at 25,000, 5,000, and 5,000 reviews, respectively.

Transformer checkpoints and Fusion pooling and regularization settings are selected using validation macro-F1. Fusion combines contextual embeddings with review-length features and optional metadata when corresponding columns are present. Ratings and sentiment labels are not model inputs.
Outputs
Each run creates a separate directory containing test metrics, class-level reports, confusion matrices, predictions, split manifests, trained models, and configuration and package-version records. Metrics are stored on their original scales, not as percentages.
Data and Availability
Dataset 1 is available from Kaggle, whose listing identifies the license as CC0 1.0 Universal (checked 25 October 2026). Download the data separately.

The distributed notebook contains no review records, personal file paths, credentials, or saved execution outputs. It does not audit free-text reviews for personally identifiable information.
Scope
This release contains the Dataset 1 main workflow. Dataset 2 and Dataset 3 pipelines and supplementary reviewer analyses are not included. Exact reproduction of the manuscript's original numerical results has not been established. Syntax and synthetic preprocessing and metric checks were completed; the packaged notebook has not undergone a full GPU training run.

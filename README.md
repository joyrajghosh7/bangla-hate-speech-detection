# Bangla Hate Speech Detection Using Machine Learning

## Project Overview
This research compares traditional machine learning models for detecting hate speech in Bengali text using character-level TF-IDF features.

## Objectives
- Classify Bengali text as hate speech or non-hate speech.
- Compare Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (Linear SVM).
- Evaluate model performance using precision, recall, F1-score, accuracy, and a confusion matrix.

## Dataset
- **Dataset:** Bengali Hate Speech Dataset
- **Source:** Kaggle
- **Original dataset size:** 30,000 records
- **Dataset link:** https://www.kaggle.com/datasets/naurosromim/bengali-hate-speech-dataset

The dataset is not included in this repository. The experiments removed 54 exact duplicate rows, 103 repeated sentences, and 21 sentences with conflicting labels, leaving 29,822 unique sentences.

## Methodology
1. Data cleaning and duplicate handling
2. Stratified 80/20 train-test split (random seed 42; 23,857 train / 5,965 test)
3. Character-level TF-IDF features: n-grams of length 2-5, 100,000 features, sublinear TF
4. Training of three models with scikit-learn default hyperparameters (balanced class weights for Logistic Regression and Linear SVM)
5. Evaluation using classification metrics and a confusion matrix

## Results (test set, n = 5,965)

| Model | Accuracy | Hate Precision | Hate Recall | Hate F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 89.14% | 84.04% | 83.20% | 83.62% | 87.75% |
| Logistic Regression | 88.99% | 83.32% | 83.70% | 83.51% | 87.62% |
| Multinomial Naive Bayes | 84.83% | 83.32% | 68.11% | 74.95% | 82.04% |

Linear SVM achieved the highest scores on the selected split, but its lead over Logistic Regression is small (0.13 points of macro F1).

## How to Run
1. Download the dataset from the Kaggle link above.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Open `bangla_hate_speech.ipynb` in Jupyter or Google Colab.
4. Set the CSV path in the first cell, then choose Run all.

## Repository Files
- `bangla_hate_speech.ipynb`: full experiment notebook
- `results/final_model_comparison.csv`: saved model comparison
- `results/linear_svm_confusion_matrix.png`: confusion matrix of the best model
- `requirements.txt`: Python dependencies

## Research Paper
[Read the research paper](https://docs.google.com/document/d/1RbBjAL_rKaIH7ks54u1EeiiDo3VASPht/edit?usp=sharing&ouid=101267860938522787154&rtpof=true&sd=true)

## Limitations
This study uses a single dataset and a single train-test split, with no hyperparameter tuning. The results do not establish generalization to all Bengali social media content. Human review is important before using predictions for moderation decisions.

## Author
[Joy Raj Ghosh], [premier university chittagong]
Research Track, Authentic Four Internship

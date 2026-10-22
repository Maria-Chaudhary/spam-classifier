# Spam Message Classifier

Binary text classifier (spam vs ham) on the **SMS Spam Collection** dataset, built with Python, scikit-learn, Pandas, Matplotlib and Seaborn.

## Pipeline
1. **Preprocessing:** lowercase → normalise URLs / emails / phone numbers / numbers / currency → regex tokenization → stop-word removal
2. **Vectorization:** Count and TF-IDF (unigrams + bigrams, `min_df=2`)
3. **Models compared:** Multinomial Naive Bayes vs Logistic Regression
4. **Selection:** 5-fold stratified CV on the training set only
5. **Precision tuning:** decision threshold chosen on out-of-fold training predictions to reach ≥ 99% precision
6. **Evaluation:** single pass on a 20 % stratified held-out test set (1034 messages)

## Data-leakage controls
- Train/test split happens before any fitting; test data is only used for final reporting.
- Vectorizer is part of the `Pipeline`, so vocabulary/IDF are learned from training folds only.
- Threshold tuning uses cross-validated predictions on training data only.
- Exact duplicate messages were removed before splitting.

## Cross-validation results (training data)
| Vectorizer | Model | CV Precision | CV Recall | CV F1 |
|---|---|---|---|---|
| TF-IDF | LogisticRegression | 0.9576 | 0.9386 | 0.9477 |
| Count | LogisticRegression | 0.9698 | 0.9080 | 0.9376 |
| Count | MultinomialNB | 0.9351 | 0.9386 | 0.9368 |
| TF-IDF | MultinomialNB | 0.9615 | 0.9099 | 0.9350 |

## Held-out test results
| Model | Vectorizer | Threshold | Accuracy | Precision | Recall | F1 | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|---|
| MultinomialNB | Count | default (0.50) | 0.9816 | 0.9242 | 0.9313 | 0.9278 | 10 | 9 |
| MultinomialNB | Count | tuned (0.997) | 0.9865 | 1.0000 | 0.8931 | 0.9435 | 0 | 14 |
| LogisticRegression | TF-IDF | default (0.50) | 0.9894 | 0.9762 | 0.9389 | 0.9572 | 3 | 8 |
| LogisticRegression | TF-IDF | tuned (0.911) | 0.9797 | 1.0000 | 0.8397 | 0.9129 | 0 | 21 |

**Final model:** Count + MultinomialNB (threshold = 0.997)

```
              precision    recall  f1-score   support

         ham     0.9847    1.0000    0.9923       903
        spam     1.0000    0.8931    0.9435       131

    accuracy                         0.9865      1034
   macro avg     0.9924    0.9466    0.9679      1034
weighted avg     0.9867    0.9865    0.9861      1034

```

Confusion matrices: `artifacts/confusion_matrices.png`, `artifacts/final_cm_and_pr_curve.png`

## Sample prediction log
| Prediction | Spam prob. | Message |
|---|---|---|
| 🚨 SPAM | 1.000 | Congratulations! You've won a £1000 Walmart gift card. Call 09061701461 now to claim  |
| 🚨 SPAM | 1.000 | URGENT! Your account has been suspended. Click http://bit.ly/secure-login to verify i |
| 🚨 SPAM | 1.000 | Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 to |
| 🚨 SPAM | 1.000 | WINNER!! As a valued network customer you have been selected to receive a £900 prize  |
| ✅ HAM | 0.000 | Hey, are we still meeting for lunch tomorrow at 1? |
| ✅ HAM | 0.008 | Can you send me the notes from today's lecture? Thanks! |
| ✅ HAM | 0.000 | Mom said dinner is ready, come home soon |
| ✅ HAM | 0.000 | I'll be late, stuck in traffic. Start without me |
| ✅ HAM | 0.029 | Your OTP for the library portal is 482913. Do not share it with anyone. |
| ✅ HAM | 0.003 | Reminder: project submission deadline is Friday 5 PM |

## Files
| File | Description |
|---|---|
| `spam_classifier.ipynb` | Full training + evaluation notebook |
| `spam_utils.py` | Preprocessing / tokenizer (needed to load the model) |
| `artifacts/spam_classifier.joblib` | Saved pipeline (vectorizer + model) and threshold |
| `artifacts/test_metrics.csv` | Test metrics table |

## Usage
```python
import joblib, spam_utils            # spam_utils must be importable
b = joblib.load("artifacts/spam_classifier.joblib")
prob = b["pipeline"].predict_proba(["Win a free prize now!"])[0, 1]
print("SPAM" if prob >= b["threshold"] else "HAM", prob)
```

## Run
```
pip install pandas scikit-learn nltk matplotlib seaborn joblib tabulate
jupyter notebook spam_classifier.ipynb   # Run All
```

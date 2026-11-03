# CENG313 Data Science Project - Eedi: Mining Misconceptions in Mathematics

## Team Members
| Şevket Berat Tetik | 23118080062 | 

| Alperen örene | 23118080006|

## 1. Competition & Dataset Overview
* **Kaggle Competition:** [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
* **Problem Description:** Predicting the specific mathematical misconception associated with each incorrect distractor option in multiple-choice diagnostic questions.
* **Dataset Files:**
  * `train.csv`: Diagnostic questions, 4 answer choices (A-D), correct answer label, and target `MisconceptionId` for each distractor.
  * `test.csv`: Unlabeled test questions for generating top-25 misconception predictions.
  * `misconception_mapping.csv`: Mapping of 2,587 unique `MisconceptionId` values to their natural language descriptions (`MisconceptionName`).
  * `sample_submission.csv`: Required submission format evaluated with **MAP@25**.

## 2. Project Structure
GROUP24_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
└── src/
    ├── data.py
    ├── models.py
    └── utils.py

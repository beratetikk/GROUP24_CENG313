# CENG313 Data Science Project - Eedi: Mining Misconceptions in Mathematics

## Team Members
* Şevket Berat Tetik 23118080062 @beratetikk
* Alperen Örene 23118080062 @Alperenorene

## 1. Competition & Dataset Overview
* **Kaggle Competition:** [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
* **Problem Description:** Predicting the specific mathematical misconception associated with each incorrect distractor option in multiple-choice diagnostic questions.
* **Dataset Files:**
  * `train.csv`: Diagnostic questions, 4 answer choices (A-D), correct answer label, and target `MisconceptionId` for each distractor.
  * `test.csv`: Unlabeled test questions for generating top-25 misconception predictions.
  * `misconception_mapping.csv`: Mapping of 2,587 unique `MisconceptionId` values to their natural language descriptions (`MisconceptionName`).
  * `sample_submission.csv`: Required submission format evaluated with **MAP@25**.

## 2. Project Structure
```text
GROUP24_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
└── src/
    ├── data.py
    ├── models.py
    └── utils.py
```

## 3. Approach & Methodology
* **Assignment 1 (EDA & Baselines):** Exploratory Data Analysis on question/distractor distributions, missing target analysis, and text length statistics. Implementation of TF-IDF + Cosine Similarity and CountVectorizer (Bag-of-Words) baseline retrieval models.
* **Assignment 2 (Proposed Solution):** *will be updated*

## 4. Results Comparison
| Model | Validation MAP@25 | Kaggle Public Score |
| :--- | :---: | :---: |
| Baseline 1 (TF-IDF + Cosine Similarity) | TBD | TBD |
| Baseline 2 (BoW / CountVectorizer) | TBD | TBD |
| Proposed Solution (Assignment 2) | TBD | TBD |

## 5. How to Run the Notebooks
1. Open `assignment1_eda_baseline.ipynb` or `assignment2_solution.ipynb` in Google Colab.
2. Ensure the `src/` directory and Kaggle CSV files are in the working directory.
3. Run all cells sequentially from top to bottom.

## 6. Progress Logs
* **Step 1:** Repository created, initial modular folder structure (`src/`) and `README.md` initialized.
* **Step 2:** Implemented data loading and wide-to-long distractor formatting in `src/data.py`.

## 7. Kaggle Submission History
| Date | Submission File | Description | Public Score |
| :--- | :--- | :--- | :---: |
| TBD | `submission_baseline.csv` | Assignment 1 TF-IDF Baseline | TBD |

import os
import pandas as pd
import numpy as np


def load_datasets(data_dir="."):
    """
    4 ana CSV dosyasini okur
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    mapping_df = pd.read_csv(os.path.join(data_dir, "misconception_mapping.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train_df, test_df, mapping_df, sub_df


def get_missing_summary(df):
    """
    Veri setindeki eksik (NaN) degerlerin sayisini ve yuzdesini hesaplar
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_pct.round(2)
    })
    return summary[summary["Missing Count"] > 0].sort_values(
        by="Missing Count", ascending=False
    )


def convert_wide_to_long(df, is_train=True):
    """
    Her soruda dogru olmayan 3 sikki ayri satirlara donusturup mdoel icin full_text olusturur 
    """
    records = []
    options = ["A", "B", "C", "D"]

    for _, row in df.iterrows():
        q_id = row["QuestionId"]
        correct_opt = row["CorrectAnswer"]
        correct_text = str(row[f"Answer{correct_opt}Text"])
        
        for opt in options:
            if opt == correct_opt:
                continue

            ans_text = str(row[f"Answer{opt}Text"])
            full_text = (
                f"Subject: {row['SubjectName']} | "
                f"Construct: {row['ConstructName']} | "
                f"Question: {row['QuestionText']} | "
                f"Correct Answer: {correct_text} | "
                f"Incorrect Answer ({opt}): {ans_text}"
            )

            item = {
                "QuestionId_Answer": f"{q_id}_{opt}",
                "QuestionId": q_id,
                "ConstructId": row["ConstructId"],
                "ConstructName": row["ConstructName"],
                "SubjectId": row["SubjectId"],
                "SubjectName": row["SubjectName"],
                "QuestionText": row["QuestionText"],
                "CorrectAnswer": correct_opt,
                "CorrectAnswerText": correct_text,
                "DistractorOption": opt,
                "DistractorText": ans_text,
                "full_text": full_text,
            }

            if is_train:
                item["MisconceptionId"] = row[f"Misconception{opt}Id"]

            records.append(item)

    return pd.DataFrame(records)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#grafik için tema ayari
sns.set_theme(style="whitegrid")


def plot_top_categories(df, column, top_n=15, title="Top Categories", xlabel="Count"):
    """
    Verilen sutundaki en cok tekrar eden ilk N kategoriyi yatay cubuk grafik olarak cizer.
    """
    plt.figure(figsize=(10, 6))
    counts = df[column].value_counts().head(top_n)
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_text_length_distribution(df, column, title="Text Word Count Distribution"):
    """
    Metinlerin kelime sayilarini hesaplar ve histogramini cizer.
    """
    plt.figure(figsize=(9, 5))
    word_counts = df[column].astype(str).apply(lambda x: len(x.split()))
    sns.histplot(word_counts, bins=30, kde=True, color="royalblue")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Kelime Sayisi (Word Count)", fontsize=12)
    plt.ylabel("Frekans (Soru Sayisi)", fontsize=12)
    plt.tight_layout()
    plt.show()


def apk(actual, predicted, k=25):
    """
    Tek bir satir icin Average Precision @ k (AP@25) hesaplar.
    Dogru tahmin ilk siradaysa 1.0 puan, 2. siradaysa 0.5 puan, 3. siradaysa 0.33 puan verir.
    """
    if actual in predicted[:k]:
        index = predicted[:k].index(actual)
        return 1.0 / (index + 1.0)
    return 0.0


def mapk(actual_list, predicted_list, k=25):
    """
    Tum tahminler icin Mean Average Precision @ k (MAP@25) ortalamasini hesaplar.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual_list, predicted_list)])

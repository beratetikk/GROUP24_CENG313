#temel model ve onerilen model sınıfları
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MostFrequentBaseline:
    """
    Egitim setindeki en sik gecen 25 kavram yanilgisini tahmin eder
    """
    def _init_(self, k=25):
        self.k = k
        self.top_k_ids = []

    def fit(self, train_df, target_col="MisconceptionId"):
        self.top_k_ids = (
            train_df[target_col]
            .value_counts()
            .head(self.k)
            .index.astype(int)
            .tolist()
        )
        return self

    def predict(self, df):
        return [self.top_k_ids for _ in range(len(df))]


class TextSimilarityBaseline:
    """
    Soru metinleri (full_text) ile kavram yanilgisi tanimlari (MisconceptionName)
    arasindaki metin benzerligini (Cosine Similarity) hesaplayarak ilk 25 tahmini siralar.
    """
    def _init_(self, method="tfidf", k=25):
        self.k = k
        self.method = method
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
        self.misconception_vectors = None
        self.misconception_ids = None

    def fit(self, mapping_df):
        self.misconception_ids = mapping_df["MisconceptionId"].astype(int).values
        texts = mapping_df["MisconceptionName"].astype(str).tolist()
        self.misconception_vectors = self.vectorizer.fit_transform(texts)
        return self

    def predict(self, df, text_col="full_text"):
        query_texts = df[text_col].astype(str).tolist()
        query_vectors = self.vectorizer.transform(query_texts)

        sim_matrix = cosine_similarity(query_vectors, self.misconception_vectors)
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :self.k]

        predictions = []
        for idx_row in top_k_indices:
            predictions.append(self.misconception_ids[idx_row].tolist())
        return predictions
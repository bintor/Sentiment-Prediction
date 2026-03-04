import os
import joblib
import pandas as pd

MODEL_PATH = "models"


class SVMService:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.preprocessor = None

    def load(self):
        """Load semua komponen SVM"""
        self.model = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
        self.tfidf = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
        self.label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
        self.preprocessor = joblib.load(os.path.join(MODEL_PATH, "preprocessor.pkl"))
        return self

    def predict_dataframe(self, df: pd.DataFrame, text_col: str):
        df = df.copy()

        df["cleaned"] = df[text_col].apply(self.preprocessor.preprocess)
        vectors = self.tfidf.transform(df["cleaned"])

        preds = self.model.predict(vectors)
        df["sentiment"] = self.label_encoder.inverse_transform(preds)

        if hasattr(self.model, "decision_function"):
            logits = self.model.decision_function(vectors)

            if logits.ndim == 1:
                logits = logits.reshape(-1, 1)

            df["logits"] = logits.tolist()
        else:
            df["logits"] = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectors)
            df["confidence"] = [round(max(p) * 100, 2) for p in probs]

            df["probabilities"] = probs.tolist()
        else:
            df["confidence"] = 100.0
            df["probabilities"] = None

        return df


    def predict_text(self, text: str):
        cleaned = self.preprocessor.preprocess(text)
        vector = self.tfidf.transform([cleaned])

        pred = self.model.predict(vector)[0]
        label = self.label_encoder.inverse_transform([pred])[0]

        logits = None
        if hasattr(self.model, "decision_function"):
            logit = self.model.decision_function(vector)

            if logit.ndim == 1:
                logit = logit.reshape(1, -1)

            logits = logit[0].tolist()

        confidence = None
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(vector)[0]
            confidence = round(max(prob) * 100, 2)
            probabilities = prob.tolist()

        return {
            "sentiment": label,
            "confidence": confidence,
            "logits": logits,
            "probabilities": probabilities,
        }


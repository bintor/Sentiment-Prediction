import torch
import pickle
import os
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_DIR = "models/saved_models"


class IndoBERTClassifier(nn.Module):
    def __init__(self, num_labels: int, activation="none"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)

        hidden_size = self.bert.config.hidden_size

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "normtanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        if self.activation:
            pooled = self.activation(pooled)

        return self.classifier(pooled)


class IndoBERTService:
    def __init__(self, variant="gelu"):
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        self.num_labels = len(self.label_encoder.classes_)

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_DIR, "tokenizer")
        )

        self.model = IndoBERTClassifier(
            self.num_labels,
            activation=variant
        )

        model_path = os.path.join(
            MODEL_DIR, f"indobert_{variant}.pt"
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

    def predict_dataframe(self, df: pd.DataFrame, text_col: str):
        texts = df[text_col].astype(str).tolist()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

        df = df.copy()
        df["sentiment"] = self.label_encoder.inverse_transform(preds)
        df["confidence"] = (confidences * 100).round(2)

        return df
    def predict_dataframe(self, df: pd.DataFrame, text_col: str):
        texts = df[text_col].astype(str).tolist()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            confidences = probs.max(dim=1).values

        df = df.copy()

        df["sentiment"] = self.label_encoder.inverse_transform(
            preds.cpu().numpy()
        )

        df["confidence"] = (confidences.cpu().numpy() * 100).round(2)

        df["logits"] = logits.cpu().numpy().tolist()

        df["probabilities"] = probs.cpu().numpy().tolist()

        return df
    
    def entropy_from_probs(probs):
        probs = np.array(probs)
        return -np.sum(probs * np.log(probs + 1e-9), axis=1)

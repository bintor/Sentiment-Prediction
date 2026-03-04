import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
from io import BytesIO
import re
import json
import ast
import string

from scraper import get_twitter_data
from database import (
    init_db,
    save_analysis,
    get_history,
    get_detail,
    export_database,
    migrate_add_entropy
)

from ml.svm_service import SVMService
from ml.indobert_service import IndoBERTService

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set([
            'yang','untuk','pada','ke','para','namun','menurut','antara',
            'dia','dua','ia','seperti','jika','sehingga','kembali',
            'dan','ini','itu','atau','adalah','with','dari','di',
            'akan','oleh','tidak','telah','dalam','bisa','ada','juga',
            'dapat','sudah','saya','ya','aja','nya','si','lagi'
        ])

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def preprocess(self, text):
        return ' '.join(
            w for w in self.clean_text(text).split()
            if w not in self.stopwords
        )

load_dotenv()
init_db()
migrate_add_entropy()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")

st.set_page_config(
    page_title="Sentiment Prediction System",
    layout="wide",
)


st.markdown("""
<style>
.main { background-color: #f8f9fa; }

.sentiment-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.pos { border-left: 8px solid #2196F3; }
.neg { border-left: 8px solid #F44336; }
.neu { border-left: 8px solid #9E9E9E; }

.card-meta {
    font-size: 0.85rem;
    font-weight: 700;
    color: #757575;
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
    border-bottom: 1px solid #f0f0f0;
    padding-bottom: 5px;
}

.card-text {
    color: #212121;
    line-height: 1.6;
    font-size: 1rem;
}

div.stButton > button:first-child {
    background-color: #757575;
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_svm_service():
    return SVMService().load()

@st.cache_resource
def get_indobert_service(variant: str):
    return IndoBERTService(variant)

ALL_MODELS = [
    "SVM Political Classifier",
    "IndoBERT (none)",
    "IndoBERT (gelu)",
    "IndoBERT (normtanh)",
]

COLOR_MAP = {
    'positive': '#2196F3',
    'negative': '#F44336',
    'neutral': '#9E9E9E',
    'Positif': '#2196F3',
    'Negatif': '#F44336',
    'Netral': '#9E9E9E',
}


def calculate_entropy(probabilities):
    probs = np.array(probabilities)
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)

def run_model(df, text_col, model_name):
    if model_name == "SVM Political Classifier":
        out = get_svm_service().predict_dataframe(df, text_col)
    else:
        variant = model_name.lower().split("(")[1].replace(")", "").strip()
        out = get_indobert_service(variant).predict_dataframe(df, text_col)

    if "probabilities" in out.columns and out["probabilities"].notnull().all():
        out["entropy"] = calculate_entropy(out["probabilities"].tolist())
    else:
        out["entropy"] = 0.0 

    return out


def display_visuals(df, title):
    st.subheader(title)

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            df,
            names="sentiment",
            hole=0.5,
            color="sentiment",
            color_discrete_map=COLOR_MAP,
            title="Sentiment Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        avg = df.groupby("sentiment")["confidence"].mean().reset_index()
        fig_bar = px.bar(
            avg,
            x="sentiment",
            y="confidence",
            color="sentiment",
            color_discrete_map=COLOR_MAP,
            title="Average Confidence",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    if "entropy" in df.columns:
        st.subheader("Prediction Uncertainty (Entropy)")
        fig_entropy = px.histogram(
            df,
            x="entropy",
            nbins=30,
            title="Entropy Distribution (Higher = More Uncertain)",
        )
        st.plotly_chart(fig_entropy, use_container_width=True)

# =========================
# CARDS
# =========================
def render_sentiment_cards(df, text_col):
    for _, row in df.iterrows():
        sentiment = str(row["sentiment"]).upper()
        cls = "pos" if "POS" in sentiment else "neg" if "NEG" in sentiment else "neu"

        entropy_text = (
            f" | ENTROPY: {round(row['entropy'], 4)}"
            if row.get("entropy") is not None
            else ""
        )

        st.markdown(f"""
        <div class="sentiment-card {cls}">
            <div class="card-meta">
                <span>{sentiment}</span>
                <span>
                    CONFIDENCE: {row['confidence']}%
                    {entropy_text}
                </span>
            </div>
            <div class="card-text">{row[text_col]}</div>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Menu")
    menu = st.radio(
        "Select Menu",
        ["Live Analysis", "History Logs", "Model Evaluation"],
        index=0
    )

    st.divider()

    selected_model = st.selectbox(
        "Intelligence Model",
        ALL_MODELS + ["All Models (Comparison)"]
    )

if menu == "Live Analysis":
    st.title("Sentiment Live Analysis")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        query = st.text_input("Monitoring Topic", placeholder="Contoh: Prabowo Gibran")
    with col_b:
        limit = st.number_input("Data Limit", 10, 500, 30)

    if st.button("Execute Analysis"):
        with st.spinner("Processing..."):
            df = get_twitter_data(APIFY_TOKEN, query, limit)

            if df.empty:
                st.warning("No data found.")
            else:
                if selected_model != "All Models (Comparison)":
                    out = run_model(df, "text", selected_model)
                    save_analysis(query, selected_model, out)
                    display_visuals(out, selected_model)
                    render_sentiment_cards(out, "text")

                    with st.expander("Raw Prediction Data (Logits & Entropy)"):
                        st.dataframe(
                            out[[
                                "text",
                                "sentiment",
                                "confidence",
                                "entropy",
                                "logits",
                                "probabilities"
                            ]],
                            use_container_width=True
                        )

                else:
                    tabs = st.tabs(ALL_MODELS)
                    for model, tab in zip(ALL_MODELS, tabs):
                        with tab:
                            out = run_model(df.copy(), "text", model)
                            display_visuals(out, model)
                            render_sentiment_cards(out, "text")


elif menu == "History Logs":
    st.title("Analysis Records")

    hist = get_history()
    if not hist.empty:
        st.dataframe(hist, use_container_width=True, hide_index=True)
        sid = st.selectbox("Select History ID", hist.id)

        raw = get_detail(sid)
        re_model = st.selectbox("Re-analyze With", ALL_MODELS)

        if st.button("Start Re-analysis"):
            out = run_model(raw.copy(), "text", re_model)
            display_visuals(out, re_model)
            render_sentiment_cards(out, "text")
            
            with st.expander("Raw Prediction Data (Logits & Entropy)"):
                        st.dataframe(
                            out[[
                                "text",
                                "sentiment",
                                "confidence",
                                "entropy",
                                "logits",
                                "probabilities"
                            ]],
                            use_container_width=True
                        )


elif menu == "Model Evaluation":
    st.title("Performance Metrics")

    uploaded_file = st.file_uploader(
        "Upload labeled CSV (columns: text, sentiment)",
        type=["csv"]
    )

    eval_model = st.selectbox("Evaluation Model", ALL_MODELS)

    if uploaded_file:
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        import matplotlib.pyplot as plt

        df_test = pd.read_csv(uploaded_file)

        if st.button("Run Evaluation"):
            pred = run_model(df_test.copy(), "text", eval_model)
            st.text(classification_report(df_test["sentiment"], pred.sentiment))

            fig, ax = plt.subplots()
            sns.heatmap(
                confusion_matrix(df_test["sentiment"], pred.sentiment),
                annot=True,
                fmt="d",
                cmap="Greys",
                ax=ax
            )
            st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.subheader("Database")


from io import BytesIO


def extract_username(author):
    """
    author bisa:
    - dict
    - JSON string
    - string repr dict
    - None
    """
    if author is None:
        return ""

    if isinstance(author, dict):
        return author.get("userName", "")

    if isinstance(author, str):
        author = author.strip()

        try:
            data = json.loads(author)
            if isinstance(data, dict):
                return data.get("userName", "")
        except Exception:
            pass

        try:
            data = ast.literal_eval(author)
            if isinstance(data, dict):
                return data.get("userName", "")
        except Exception:
            pass

    return ""


if st.sidebar.button("Export Database (XLSX)"):
    df_export = export_database()

    if df_export.empty:
        st.sidebar.warning("Database masih kosong.")
    else:
        if "author" in df_export.columns:
            df_export = df_export.copy()
            df_export["author"] = df_export["author"].apply(extract_username)

        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_export.to_excel(
                writer,
                index=False,
                sheet_name="Sentiment Analysis"
            )

            # (OPSIONAL) auto column width
            worksheet = writer.sheets["Sentiment Analysis"]
            for idx, col in enumerate(df_export.columns):
                max_len = max(
                    df_export[col].astype(str).map(len).max(),
                    len(col)
                )
                worksheet.set_column(idx, idx, min(max_len + 2, 40))

        output.seek(0)

        st.sidebar.download_button(
            label="Download Excel",
            data=output,
            file_name="sentiment_analysis_database.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "analysis_history.db"

CONFIDENCE_THRESHOLD = 70.0 

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            keyword TEXT,
            model TEXT,
            total_data INTEGER,
            pos_count INTEGER,
            neg_count INTEGER,
            neu_count INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS results_detail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER,
            text TEXT,
            sentiment TEXT,
            confidence REAL,
            entropy REAL,
            author TEXT,
            created_at TEXT,
            url TEXT,
            retweets INTEGER,
            replies INTEGER,
            likes INTEGER,
            quotes INTEGER,
            views INTEGER,
            FOREIGN KEY(history_id) REFERENCES history(id)
        )
    """)

    conn.commit()
    conn.close()



def save_analysis(keyword, model, df):
    conn = sqlite3.connect(DB_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = conn.cursor()

    df_filtered = df[df["confidence"] >= CONFIDENCE_THRESHOLD]

    counts = df_filtered["sentiment"].value_counts()
    pos = int(counts.get("positive", counts.get("Positif", 0)))
    neg = int(counts.get("negative", counts.get("Negatif", 0)))
    neu = int(counts.get("neutral", counts.get("Netral", 0)))

    c.execute("""
        INSERT INTO history
        (timestamp, keyword, model, total_data, pos_count, neg_count, neu_count)
        VALUES (?,?,?,?,?,?,?)
    """, (
        timestamp,
        keyword,
        model,
        len(df_filtered),
        pos,
        neg,
        neu
    ))

    history_id = c.lastrowid

    for _, row in df_filtered.iterrows():
        c.execute("""
            INSERT INTO results_detail (
                history_id,
                text,
                sentiment,
                confidence,
                entropy,
                author,
                created_at,
                url,
                retweets,
                replies,
                likes,
                quotes,
                views
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            history_id,
            str(row.get("text", "")),
            str(row["sentiment"]),
            float(row["confidence"]),
            float(row.get("entropy", 0.0)),
            str(row.get("author", "")),
            str(row.get("createdAt", "")),
            str(row.get("url", "")),
            int(row.get("retweets", 0)),
            int(row.get("replies", 0)),
            int(row.get("likes", 0)),
            int(row.get("quotes", 0)),
            int(row.get("views", 0)),
        ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

def get_detail(history_id):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT * FROM results_detail WHERE history_id = {history_id}", conn)
    conn.close()
    return df

def export_database():
    conn = sqlite3.connect(DB_NAME)

    query = """
    SELECT
        d.id,
        d.text,
        d.sentiment,
        d.confidence,
        d.entropy,
        d.author,
        d.created_at,
        d.url,
        d.retweets,
        d.replies,
        d.likes,
        d.quotes,
        d.views
    FROM results_detail d
    ORDER BY d.id DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def migrate_add_entropy():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    columns = [row[1] for row in c.execute("PRAGMA table_info(results_detail)")]

    if "entropy" not in columns:
        c.execute("ALTER TABLE results_detail ADD COLUMN entropy REAL")

    conn.commit()
    conn.close()

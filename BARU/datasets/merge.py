import os
import json
import pandas as pd

folder_path = r"G:\kerja\tajoki\python\analisis-streamlit-v2\datasets\json"
all_texts = []

def extract_text(obj):
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            all_texts.append({"text": obj["text"]})
        for v in obj.values():
            extract_text(v)
    elif isinstance(obj, list):
        for item in obj:
            extract_text(item)

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                extract_text(data)
            except Exception as e:
                print(f"⚠️ Error di file {filename}: {e}")

df = pd.DataFrame(all_texts).drop_duplicates()
df.to_csv("./datasets/tweets_texts.csv", index=False, encoding="utf-8-sig")

print(f"✅ Selesai! Tersimpan {len(df)} tweet di tweets_texts.csv")

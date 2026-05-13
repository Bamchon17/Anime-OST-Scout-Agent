import kagglehub
import pandas as pd
import os

# 1. ดาวน์โหลด Dataset
path = kagglehub.dataset_download("divyanshusingh369/aimi-anime-rag-and-receipts-dataset-sample")
print("Path to dataset files:", path)

# 2. ระบุชื่อไฟล์
file_name = 'anime_dataset_small_nomic.parquet'
file_path = os.path.join(path, file_name)

# 3. โหลดไฟล์และเซฟเป็น JSON
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print("\n--- Success! Dataset Loaded ---")

    # --- ส่วนที่เพิ่มเข้ามาเพื่อเซฟเป็น JSON ---
    json_file_name = "General_meta.json"
    
    # orient='records' จะทำให้ได้ format แบบ [ {"col1": "val1"}, {"col1": "val2"} ] ซึ่งเหมาะกับงาน RAG/API
    # force_ascii=False เพื่อให้รองรับภาษาไทยหรือตัวอักษรพิเศษ (ถ้ามี)
    df.to_json(json_file_name, orient='records', force_ascii=False, indent=4)
    
    print(f"--- Data saved to {json_file_name} successfully! ---")
    # ---------------------------------------

    print(f"Total records: {len(df)}")
    print("\n--- Columns in this dataset ---")
    for i, col in enumerate(df.columns.tolist()):
        print(f"{i+1}. {col}")
else:
    print(f"Error: File {file_name} not found at {path}")
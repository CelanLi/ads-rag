import os
import json

def export_json(output_dir: str, file_name: str, content: dict):
    os.makedirs(output_dir, exist_ok=True)
    json_file = os.path.join(output_dir, f"{file_name}.json")
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(content, jf, ensure_ascii=False, indent=2)

def export_txt(output_dir: str, file_name: str, content: str):
    os.makedirs(output_dir, exist_ok=True)
    text_file = os.path.join(output_dir, f"{file_name}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(content)
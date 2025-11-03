import os
import json
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and PIL Images."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.integer,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif PIL_AVAILABLE and isinstance(obj, Image.Image):
            # PIL Images cannot be serialized to JSON
            # Return a string representation instead
            return f"<PIL.Image.Image image mode={obj.mode} size={obj.size}>"
        return super().default(obj)


def export_json(output_dir: str, file_name: str, content: dict):
    os.makedirs(output_dir, exist_ok=True)
    json_file = os.path.join(output_dir, f"{file_name}.json")
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(content, jf, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def export_txt(output_dir: str, file_name: str, content: str):
    os.makedirs(output_dir, exist_ok=True)
    text_file = os.path.join(output_dir, f"{file_name}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(content)

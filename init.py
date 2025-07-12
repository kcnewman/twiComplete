import os
from pathlib import Path

project_name = "twiComplete"

files = [
    "./data/.gitkeep",
    "./preprocess.py",
    "./ngram_model.py",
    "./autocomplete.py",
    "./evaluate.py",
    "./main.py",
]

for filePath in files:
    filePath = Path(filePath)
    fileDir, fileName = os.path.split(str(filePath))
    if fileDir != "":
        os.makedirs(fileDir, exist_ok=True)
    if not filePath.exists():
        with open(filePath, "w") as f:
            pass

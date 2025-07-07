"""
Genera dynamics_sequences.csv con las rutas a los .npy generados.
"""
import pandas as pd
from pathlib import Path
from pyprojroot import here

ROOT      = here()
FEAT_DIR  = ROOT / "data" / "dynamics" / "features"
CSV_PATH  = ROOT / "data" / "dynamics_sequences.csv"

records = []
for letter_dir in sorted(FEAT_DIR.iterdir()):
    if not letter_dir.is_dir():
        continue
    label = letter_dir.name
    for npy in sorted(letter_dir.glob("*.npy")):
        rel_path = npy.relative_to(ROOT).as_posix()
        records.append({"feature_path": rel_path, "label": label})

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)
print("✅ CSV guardado en", CSV_PATH)
print(df.head())

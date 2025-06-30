# Lee los clips extraídos y crea un CSV con paths de secuencia de frames y sus etiquetas.
from pyprojroot import here
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_DIR = here() / "data" / "letters" / "dynamics"
data = []

# Recorremos cada letra (J, Ñ, etc.)
for letter_dir in sorted(BASE_DIR.iterdir()):
    if not letter_dir.is_dir():
        continue
    letter = letter_dir.name

    # Recorremos cada secuencia de frames (carpeta)
    for seq_folder in tqdm(sorted(letter_dir.iterdir()), desc=f"Procesando {letter}"):
        if not seq_folder.is_dir():
            continue

        # Rutas a los frames ordenadas
        frame_paths = sorted([
            f.relative_to(here()).as_posix()
            for f in seq_folder.glob("*.jpg")
        ])

        if not frame_paths:
            continue

        data.append({
            "frames": frame_paths,
            "label": letter
        })

# Guardar en CSV
df = pd.DataFrame(data)
out_path = here() / "data" / "letters" / "dynamics_sequences.csv"
df.to_csv(out_path, index=False)

print(f"\n✅ CSV de secuencias dinámicas generado: {out_path}")
print(df.head(2))

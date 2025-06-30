#Crea un CSV con image_path y label, desde statics/. Aplica limpieza, filtrado, y balanceo.

from pyprojroot import here
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

# Ruta base desde raíz del proyecto
BASE_DIR = here() / "data" / "letters" / "statics"

# Lista para guardar registros válidos
data = []

# Recorremos cada letra (carpeta)
for letra_dir in sorted(BASE_DIR.iterdir()):
    if not letra_dir.is_dir():
        continue
    letra = letra_dir.name

    for imagen_path in tqdm(list(letra_dir.glob("*")), desc=f"Procesando {letra}"):
        try:
            img = cv2.imread(str(imagen_path))
            if img is None:
                print(f"⚠️ Imagen no válida: {imagen_path}")
                continue

            # Guardar path relativo (como string con /) y etiqueta
            rel_path = imagen_path.relative_to(here()).as_posix()
            data.append({"image_path": rel_path, "label": letra})

        except Exception as e:
            print(f"❌ Error al procesar {imagen_path}: {e}")

# Guardar en CSV
df = pd.DataFrame(data)
output_csv = here() / "data" / "letters" / "letter_labels.csv"
df.to_csv(output_csv, index=False)

print(f"\n✅ CSV generado correctamente: {output_csv}")
print(df.head())
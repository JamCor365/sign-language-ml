# Para los videos dinámicos: extrae frames o vectores (usando VGG16, SVD, etc.). Genera .npy o .csv por clip.
from pyprojroot import here
from pathlib import Path
import cv2
from tqdm import tqdm

# Parámetros
INPUT_DIR = here() / "data" / "letters" / "dynamics"
FRAME_SKIP = 5  # Cada cuántos frames guardar
OUT_FOLDER = "frames"  # Subcarpeta opcional si deseas cambiar la ubicación futura

# Recorremos cada letra
for letra_dir in sorted(INPUT_DIR.iterdir()):
    if not letra_dir.is_dir():
        continue
    letra = letra_dir.name

    # Recorremos cada video .mp4
    for video_path in tqdm(letra_dir.glob("*.mp4"), desc=f"Procesando {letra}"):
        nombre_clip = video_path.stem
        out_dir = letra_dir / nombre_clip
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frame_id = 0
        saved_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:
                out_path = out_dir / f"frame_{saved_id:03}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_id += 1

            frame_id += 1

        cap.release()

print("✅ Extracción de frames completada.")
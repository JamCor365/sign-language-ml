"""
Genera cuatro archivos .npy:
  • static_X.npy / static_y.npy   (HOG para SVM)
  • dynamic_X.npy / dynamic_y.npy (clips 16×224×224 RGB para I3D-light)
"""

from pathlib import Path
from pyprojroot import here
import cv2, numpy as np
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm

# ──────────────────── CONFIG ────────────────────
ROOT          = Path(here())           # raíz del proyecto
DIR_DATOS     = ROOT / "data"
DIR_STATIC    = DIR_DATOS / "statics"
DIR_DYNAMIC   = DIR_DATOS / "dynamics"

IMG_SIZE      = 64     # para HOG
CLIP_LEN      = 16     # nº frames por clip
VID_SIZE      = 224    # tamaño para I3D-light
# ────────────────────────────────────────────────


def dataset_estatico():
    """Convierte .jpg en vectores HOG y guarda *.npy*"""
    X, y = [], []
    clases = sorted([p.name for p in DIR_STATIC.iterdir() if p.is_dir()])
    mapa   = {c: i for i, c in enumerate(clases)}
    print("Clases estáticas:", mapa)

    for c in clases:
        for img_path in tqdm((DIR_STATIC / c).glob("*.jpg"), desc=f"HOG {c}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
            img = exposure.adjust_gamma(img, 1.0)

            feat = hog(img,
                       orientations=8,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm="L2-Hys",
                       transform_sqrt=True,
                       feature_vector=True)
            X.append(feat)
            y.append(mapa[c])

    np.save(DIR_DATOS / "static_X.npy", np.asarray(X, np.float32))
    np.save(DIR_DATOS / "static_y.npy", np.asarray(y, np.int64))
    print("✅ Guardado static_X.npy / static_y.npy")


def sample_clip(cap, total_frames):
    """Devuelve clip uniforme de CLIP_LEN frames (RGB uint8) o None"""
    idxs   = np.linspace(0, total_frames - 1, CLIP_LEN, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            return None
        fr = cv2.resize(fr, (VID_SIZE, VID_SIZE), cv2.INTER_AREA)
        frames.append(fr[..., ::-1])          # BGR → RGB
    return np.stack(frames)                   # (T,H,W,C)


def dataset_dinamico():
    """Extrae clips fijos de los .mp4 y guarda *.npy*"""
    X, y = [], []
    clases = sorted([p.name for p in DIR_DYNAMIC.iterdir() if p.is_dir()])
    mapa   = {c: i for i, c in enumerate(clases)}
    print("Clases dinámicas:", mapa)

    for c in clases:
        for vid_path in tqdm((DIR_DYNAMIC / c).glob("*.mp4"), desc=f"Clips {c}"):
            cap = cv2.VideoCapture(str(vid_path))
            frames_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames_tot < CLIP_LEN:
                cap.release()
                continue
            clip = sample_clip(cap, frames_tot)
            cap.release()
            if clip is not None:
                X.append(clip.astype(np.uint8))
                y.append(mapa[c])

    np.save(DIR_DATOS / "dynamic_X.npy", np.asarray(X, np.uint8))
    np.save(DIR_DATOS / "dynamic_y.npy", np.asarray(y, np.int64))
    print("✅ Guardado dynamic_X.npy / dynamic_y.npy")


if __name__ == "__main__":
    dataset_estatico()
    dataset_dinamico()
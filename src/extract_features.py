"""
Extrae landmarks directamente de los vídeos .mp4
y guarda un .npy por clip.
"""
import cv2, sys, math
from pathlib import Path
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from pyprojroot import here

ROOT      = here()
VIDEO_DIR = ROOT / "data" / "dynamics"
OUT_DIR   = ROOT / "data" / "dynamics" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def extract_landmarks(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    xyz = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    xyz -= xyz[0]
    scale = np.linalg.norm(xyz).mean()
    if scale > 0: xyz /= scale
    return xyz.flatten()

def process_video(path):
    cap      = cv2.VideoCapture(str(path))
    lms_list = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        lm = extract_landmarks(frame)
        if lm is None:
            lm = lms_list[-1] if lms_list else np.zeros(63)
        lms_list.append(lm)
    cap.release()

    if len(lms_list) < SEQ_LEN:
        lms_list.extend([lms_list[-1]] * (SEQ_LEN - len(lms_list)))
    return np.array(lms_list[:SEQ_LEN])   # (SEQ_LEN,63)

for letter_dir in sorted(VIDEO_DIR.iterdir()):
    if not letter_dir.is_dir(): continue
    letter = letter_dir.name
    for mp4 in tqdm(letter_dir.glob("*.mp4"), desc=f"Procesando {letter}"):
        seq = process_video(mp4)
        out_path = OUT_DIR / letter / f"{mp4.stem}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, seq)

print("✅ Features listos en", OUT_DIR)

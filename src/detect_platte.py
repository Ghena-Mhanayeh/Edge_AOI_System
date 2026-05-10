# --- NEU: MKLDNN deaktivieren (wichtig für ARM / Raspberry Pi) ---
import torch
torch.backends.mkldnn.enabled = False
# Erklärung:
# → Verhindert "Illegal instruction" Fehler bei PyTorch auf ARM/Emulation


import numpy as np
from ultralytics import YOLO

# --- NEU: Globaler Cache für Modelle ---
_MODEL_CACHE = {}
# Erklärung:
# → Speichert bereits geladene Modelle im RAM
# → Verhindert, dass das Modell bei jedem Request neu geladen wird


def get_yolo_model(model_path: str):
    """
    Lädt das YOLO-Modell nur einmal und cached es.
    """
    if model_path not in _MODEL_CACHE:
        
        print(f"[INIT] Lade YOLO-Modell: {model_path}", flush=True)
        _MODEL_CACHE[model_path] = YOLO(model_path, task="segment")
        print("[INIT] YOLO-Modell geladen", flush=True)
    else:
        print("[CACHE] YOLO-Modell aus Cache verwendet", flush=True)

    return _MODEL_CACHE[model_path]


def detect_platte(image_bgr, model_path: str, conf: float = 0.25):
    """
    Erkennt die Platte im Bild und gibt die beste Maske zurück.
    """

    print("[STEP] detect_platte gestartet", flush=True)

    # --- GEÄNDERT: Modell wird jetzt aus Cache geholt ---
    model = get_yolo_model(model_path)

    # --- GEÄNDERT: kleinere Bildgröße für bessere Performance ---
    print("[STEP] YOLO predict startet", flush=True)
    r = model.predict(
        source=image_bgr,
        conf=conf,
        imgsz=320,   # wichtig: reduziert Rechenzeit
        verbose=True
    )[0]
    print("[STEP] YOLO predict fertig", flush=True)

    # --- unverändert: keine Maske gefunden ---
    if r.masks is None or r.masks.xy is None or len(r.masks.xy) == 0:
        print("[WARN] Keine Platte erkannt", flush=True)
        return None

    # --- beste Maske auswählen ---
    best_i = 0
    best_conf = None

    if r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.detach().cpu().numpy()
        if confs.size:
            best_i = int(np.argmax(confs))
            best_conf = float(confs[best_i])

    poly = r.masks.xy[best_i]

    print(f"[INFO] Beste Maske Index: {best_i}, Confidence: {best_conf}", flush=True)

    return {
        "poly": poly,
        "conf": best_conf,
        "best_i": best_i
    }
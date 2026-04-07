import numpy as np
from ultralytics import YOLO

def detect_platte(image_bgr, model_path: str, conf: float = 0.25):
    """
    Funktion 1:
    - Ruft das YOLOv8-seg Modell auf
    - Wenn 'platte' erkannt wird: liefert dict mit Polygon + Meta
    - Sonst: None

    Output dict enthält:
      {
        "poly": np.ndarray shape (N,2) in Pixelkoordinaten,
        "conf": float|None,
        "best_i": int
      }
    """
    model = YOLO(model_path)
    r = model.predict(source=image_bgr, conf=conf, verbose=False)[0]

    # Segmentation mit Polygonen: r.masks.xy ist eine Liste (pro Detektion ein Polygon)
    if r.masks is None or r.masks.xy is None or len(r.masks.xy) == 0:
        return None

    # Beste Detektion: höchste Confidence (falls vorhanden)
    best_i = 0
    best_conf = None
    if r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.detach().cpu().numpy()
        if confs.size:
            best_i = int(np.argmax(confs))
            best_conf = float(confs[best_i])

    poly = r.masks.xy[best_i]  # (N,2) float Pixelkoordinaten

    return {
        "poly": poly,
        "conf": best_conf,
        "best_i": best_i
    }
import numpy as np
import cv2

def _order_points(pts4: np.ndarray) -> np.ndarray:
    """
    Sortiert 4 Punkte in Reihenfolge:
    top-left, top-right, bottom-right, bottom-left
    """
    pts4 = np.array(pts4, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts4.sum(axis=1)
    diff = np.diff(pts4, axis=1)

    rect[0] = pts4[np.argmin(s)]      # top-left
    rect[2] = pts4[np.argmax(s)]      # bottom-right
    rect[1] = pts4[np.argmin(diff)]   # top-right
    rect[3] = pts4[np.argmax(diff)]   # bottom-left
    return rect

def compute_crop_data(detection: dict):
    """
    Funktion 2:
    - Nimmt die Daten aus Funktion 1 (Polygon)
    - Berechnet die 4 Eckdaten (corners4) für den Warp/Crop

    Input detection:
      { "poly": (N,2) }

    Output crop_data:
      {
        "corners4": np.ndarray shape (4,2) float32 (TL,TR,BR,BL)
      }
    """
    poly = detection["poly"]  # (N,2)

    # minAreaRect um Polygon -> 4 Eckpunkte
    pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)  # (4,2)

    corners4 = _order_points(box)
    return {"corners4": corners4}
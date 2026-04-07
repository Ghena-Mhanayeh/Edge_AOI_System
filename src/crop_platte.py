import numpy as np
import cv2

def crop_platte(image_bgr, crop_data: dict):
    """
    Funktion 3:
    - Nimmt das Originalbild + corners4
    - Führt Perspective Warp aus
    - Gibt ausgeschnittene, geradegezogene Platte zurück
    """
    corners4 = crop_data["corners4"].astype("float32")
    (tl, tr, br, bl) = corners4

    # Zielbreite/-höhe aus Kantenlängen schätzen
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB), 1)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB), 1)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(corners4, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxW, maxH))
    return warped
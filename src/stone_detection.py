import cv2
import numpy as np

SAT_THRESHOLD = 40
MIN_AREA = 4000
BORDER_MARGIN = 20

# HSV-Farbbereiche für die erlaubten Steinfarben
# OpenCV-H liegt in [0..179]
COLOR_RANGES = {
    "blue": [
        ((100, 80, 50), (130, 255, 255))
    ],
    "green": [
        ((40, 60, 50), (85, 255, 255))
    ],
    "yellow": [
        ((20, 80, 80), (35, 255, 255))
    ],
    "orange": [
        ((10, 100, 80), (19, 255, 255))
    ]
}


def classify_stone_color(hsv_img, contour, bbox):
    """
    Bestimmt die dominante Farbe eines Steins innerhalb seiner Kontur.

    Args:
        hsv_img: gesamtes Bild im HSV-Format
        contour: Kontur des Steins
        bbox: Bounding Box (x, y, w, h)

    Returns:
        color_name: 'blue', 'green', 'yellow', 'orange' oder 'unknown'
        color_scores: Pixelanzahl je Farbe innerhalb der Steinmaske
    """
    x, y, w, h = bbox

    # ROI ausschneiden
    hsv_roi = hsv_img[y:y+h, x:x+w]

    # Kontur auf ROI-Koordinaten umrechnen
    contour_roi = contour - np.array([[[x, y]]])

    # Maske für genau diesen Stein erzeugen
    stone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(stone_mask, [contour_roi], -1, 255, thickness=-1)

    # Optional: Rand etwas verkleinern, damit Übergänge/Hintergrund weniger stören
    kernel = np.ones((5, 5), np.uint8)
    stone_mask = cv2.erode(stone_mask, kernel, iterations=1)

    color_scores = {}

    for color_name, ranges in COLOR_RANGES.items():
        color_mask_total = np.zeros((h, w), dtype=np.uint8)

        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv_roi, lower_np, upper_np)
            color_mask_total = cv2.bitwise_or(color_mask_total, color_mask)

        # Nur Pixel zählen, die sowohl zur Farbe als auch zum Stein gehören
        overlap = cv2.bitwise_and(color_mask_total, stone_mask)
        score = cv2.countNonZero(overlap)
        color_scores[color_name] = score

    best_color = max(color_scores, key=color_scores.get)
    best_score = color_scores[best_color]

    # Sicherheitsprüfung: falls kaum passende Farbpixel gefunden wurden
    stone_pixels = cv2.countNonZero(stone_mask)
    if stone_pixels == 0:
        return "unknown", color_scores

    ratio = best_score / float(stone_pixels)

    # Schwellwert kannst du später anpassen
    if ratio < 0.20:
        return "unknown", color_scores

    return best_color, color_scores


def detect_stones_on_plate(plate_img_bgr, config=None):
    plate_h, plate_w = plate_img_bgr.shape[:2]

    hsv = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    _, mask = cv2.threshold(s_channel, SAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("output/debug_saturation_mask.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_stones = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)

        if x <= BORDER_MARGIN or y <= BORDER_MARGIN:
            continue
        if x + w >= plate_w - BORDER_MARGIN:
            continue
        if y + h >= plate_h - BORDER_MARGIN:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        color_name, color_scores = classify_stone_color(hsv, c, (x, y, w, h))

        detected_stones.append({
            "bbox": (x, y, w, h),
            "center_px": (cx, cy),
            "center_norm": (cx / plate_w, cy / plate_h),
            "area_px": area,
            "color": color_name,
            "color_scores": color_scores
        })

    return detected_stones
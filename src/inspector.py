import os
import cv2

from detect_platte import detect_platte
from compute_crop_data import compute_crop_data
from crop_platte import crop_platte
from config_loader import load_config
from stone_detection import detect_stones_on_plate
from compare_config import compare_stones_to_config
from visualize_result import draw_aoi_result


def inspect_image(image_path: str) -> dict:
    """
    Führt die komplette AOI-Prüfung für ein Bild aus
    und gibt ein standardisiertes Ergebnis-Dictionary zurück.
    """

    # --- Pfade robust aufbauen ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_path = os.path.join(project_root, "models", "best.pt")
    config_path = os.path.join(project_root, "config", "config.yaml")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    name_wo_ext, _ = os.path.splitext(filename)

    # --- Bild laden ---
    img = cv2.imread(image_path)
    if img is None:
        return {
            "status": "ERROR",
            "result": None,
            "message": f"Bild nicht lesbar: {image_path}",
            "output_image": None,
            "report": None
        }

    # 1) Platte erkennen
    plate_data = detect_platte(img, model_path)
    if plate_data is None:
        return {
            "status": "OK",
            "result": "NIO",
            "message": "Keine Platte erkannt",
            "output_image": None,
            "report": {
                "status": "NIO",
                "reason": "Keine Platte erkannt"
            }
        }

    # 2) Crop-Daten berechnen
    crop_data = compute_crop_data(plate_data)

    # 3) Platte ausschneiden / entzerren
    plate_img = crop_platte(img, crop_data)

    # 4) Konfiguration laden
    config = load_config(config_path)

    # 5) Steine erkennen
    detected_stones = detect_stones_on_plate(plate_img, config=config)

    # 6) Soll-Ist-Vergleich
    result, report = compare_stones_to_config(detected_stones, config)

    # 7) Visualisierung erzeugen
    vis = draw_aoi_result(plate_img, config, detected_stones, report)

    output_image_path = os.path.join(output_dir, f"{name_wo_ext}_result.png")
    cv2.imwrite(output_image_path, vis)

    # 8) Standardisierte Rückgabe
    return {
        "status": "OK",
        "result": result,          # "IO" oder "NIO"
        "message": "Prüfung abgeschlossen",
        "output_image": output_image_path,
        "report": report
    }
import uvicorn


def main():
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()

"""

import os
import cv2

from detect_platte import detect_platte
from compute_crop_data import compute_crop_data
from crop_platte import crop_platte

from config_loader import load_config
from stone_detection import detect_stones_on_plate
from compare_config import compare_stones_to_config
from visualize_result import draw_aoi_result


def main():
    # --- Pfade robust bauen (egal von wo du startest) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_path = os.path.join(project_root, "models", "best.pt")
    image_path = os.path.join(project_root, "input", "bd9fd6df2b-38.png")
    config_path = os.path.join(project_root, "config", "config.yaml")
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Bild nicht gefunden/lesbar: {image_path}")

    # 1) Modell aufrufen
    det = detect_platte(img, model_path=model_path, conf=0.25)
    if det is None:
        print("Keine Platte erkannt")
        return

    print("Platte erkannt ✅", "conf:", det["conf"])

    # 2) Eckdaten berechnen
    crop_data = compute_crop_data(det)
    corners4 = crop_data["corners4"]
    print("Eckdaten (TL,TR,BR,BL) Pixel:")
    for i, (x, y) in enumerate(corners4):
        print(f"  {i}: ({x:.1f}, {y:.1f})")

    # 3) Ausschneiden / Warp
    platte = crop_platte(img, crop_data)

    out_plate_path = os.path.join(out_dir, "platte_warp.png")
    ok = cv2.imwrite(out_plate_path, platte)
    if not ok:
        raise RuntimeError(f"Konnte nicht speichern: {out_plate_path}")

    print("Gespeichert:", out_plate_path)

    # 4) Config laden
    config = load_config(config_path)

    # 5) Steine erkennen
    detected_stones = detect_stones_on_plate(platte, config)

    print(f"Erkannte Steine: {len(detected_stones)}")

    for i, stone in enumerate(detected_stones):
        cx, cy = stone["center_px"]
        x_norm, y_norm = stone["center_norm"]
        color = stone.get("color", "unknown")
        print(
            f"  Stein {i}: "
            f"center_px=({cx:.1f}, {cy:.1f}), "
            f"center_norm=({x_norm:.4f}, {y_norm:.4f}), "
            f"color={color}"
        )

    # 6) Vergleich mit Config
    status, report = compare_stones_to_config(
        detected_stones=detected_stones,
        config=config,
        fail_on_extra=True
    )

    print("\n===== AOI Ergebnis =====")
    print("STATUS       :", status)
    print("Expected     :", report["expected_count"])
    print("Detected     :", report["detected_count"])
    print("Matched      :", report["matched_count"])
    print("Missing      :", len(report["missing"]))
    print("Extra        :", len(report["extra"]))
    print("Color errors :", len(report.get("color_mismatches", [])))

    if report["matches"]:
        print("\nGematchte Steine:")
        for item in report["matches"]:
            print(
                f"  id={item['id']} "
                f"target={item['target_norm']} "
                f"detected={item['detected_norm']} "
                f"expected_color={item.get('expected_color')} "
                f"detected_color={item.get('detected_color')} "
                f"color_ok={item.get('color_ok')}"
            )

    if report["missing"]:
        print("\nFehlende Steine:")
        for item in report["missing"]:
            print(
                f"  id={item['id']} "
                f"target={item['target_norm']} "
                f"expected_color={item.get('expected_color')} "
                f"tol={item['tolerance_norm']}"
            )

    if report["extra"]:
        print("\nZusätzliche erkannte Steine:")
        for item in report["extra"]:
            print(
                f"  detected_index={item['detected_index']} "
                f"detected_norm={item['detected_norm']} "
                f"detected_color={item.get('detected_color')} "
                f"bbox={item['bbox']}"
            )

    if report.get("color_mismatches"):
        print("\nFarbfehler:")
        for item in report["color_mismatches"]:
            print(
                f"  id={item['id']} "
                f"target={item['target_norm']} "
                f"detected={item['detected_norm']} "
                f"expected_color={item['expected_color']} "
                f"detected_color={item['detected_color']}"
            )

    # 7) Debug-Bild zeichnen und speichern
    vis_img = draw_aoi_result(platte, config, detected_stones, report)

    out_vis_path = os.path.join(out_dir, "aoi_result.png")
    ok = cv2.imwrite(out_vis_path, vis_img)
    if not ok:
        raise RuntimeError(f"Konnte nicht speichern: {out_vis_path}")

    print("Debug-Bild gespeichert:", out_vis_path)


if __name__ == "__main__":
    main()

    """
import os
import cv2
import yaml
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

@dataclass
class TeachInSettings:
    # reale Größen in cm
    plate_size_cm: Tuple[float, float] = (8.0, 6.0)      # (breite, hoehe)
    stone_size_cm: Tuple[float, float] = (0.8, 1.5)      # (breite, hoehe)

    # Toleranz als Anteil der Platte (0..1). Beispiel: 0.03 = 3%
    tolerance_norm: float = 0.03

    window_name: str = "Teach-In"
    out_dir: str = "config"
    out_name: str = "config.yaml"

# Erlaubte Farben für die Konfiguration
ALLOWED_COLORS = {
    "b": ("blue",   (255, 0, 0)),     # BGR für OpenCV
    "g": ("green",  (0, 255, 0)),
    "y": ("yellow", (0, 255, 255)),
    "o": ("orange", (0, 165, 255)),
}

def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Bild konnte nicht geladen werden: {image_path}")
    h, w = img.shape[:2]
    return img, w, h

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def out_path(settings: TeachInSettings) -> str:
    ensure_out_dir(settings.out_dir)
    return os.path.join(settings.out_dir, settings.out_name)

def stone_size_norm(settings: TeachInSettings) -> Dict[str, float]:
    plate_w_cm, plate_h_cm = settings.plate_size_cm
    stone_w_cm, stone_h_cm = settings.stone_size_cm

    if plate_w_cm <= 0 or plate_h_cm <= 0:
        raise ValueError("plate_size_cm muss > 0 sein")

    return {
        "w": stone_w_cm / float(plate_w_cm),  # Anteil an Plattenbreite
        "h": stone_h_cm / float(plate_h_cm),  # Anteil an Plattenhöhe
    }

def make_stone(
    stone_id: int,
    x_px: int,
    y_px: int,
    img_w: int,
    img_h: int,
    settings: TeachInSettings,
    color_name: str
) -> Dict:
    x_norm = x_px / float(img_w)
    y_norm = y_px / float(img_h)

    sz = stone_size_norm(settings)

    return {
        "id": int(stone_id),
        "position_norm": {
            "x": round(x_norm, 6),
            "y": round(y_norm, 6)
        },
        "color": color_name,
        "tolerance_norm": {
            "x": float(settings.tolerance_norm),
            "y": float(settings.tolerance_norm)
        },
        "stone_size_norm": {
            "w": round(sz["w"], 6),
            "h": round(sz["h"], 6)
        },
        "stone_size_cm": {
            "w": float(settings.stone_size_cm[0]),
            "h": float(settings.stone_size_cm[1])
        }
    }

def get_stone_draw_color(color_name: str) -> Tuple[int, int, int]:
    for _, (name, bgr) in ALLOWED_COLORS.items():
        if name == color_name:
            return bgr
    return (255, 255, 255)  # fallback = weiß

def draw_overlay(img, stones: List[Dict], img_w: int, img_h: int, current_color: str):
    display = img.copy()

    for s in stones:
        x_px = int(s["position_norm"]["x"] * img_w)
        y_px = int(s["position_norm"]["y"] * img_h)

        stone_color = s.get("color", "unknown")
        stone_bgr = get_stone_draw_color(stone_color)

        # Stein-Boundingbox in Steinfarbe
        sw_px = int(s["stone_size_norm"]["w"] * img_w)
        sh_px = int(s["stone_size_norm"]["h"] * img_h)
        cv2.rectangle(
            display,
            (x_px - sw_px // 2, y_px - sh_px // 2),
            (x_px + sw_px // 2, y_px + sh_px // 2),
            stone_bgr,
            2
        )

        # Toleranzrechteck (grün)
        tolx_px = int(s["tolerance_norm"]["x"] * img_w)
        toly_px = int(s["tolerance_norm"]["y"] * img_h)
        cv2.rectangle(
            display,
            (x_px - tolx_px, y_px - toly_px),
            (x_px + tolx_px, y_px + toly_px),
            (0, 255, 0),
            2
        )

        # Mittelpunkt + ID + Farbe
        cv2.circle(display, (x_px, y_px), 4, stone_bgr, -1)
        cv2.putText(
            display,
            f"ID {s['id']} | {stone_color}",
            (x_px + 6, y_px - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            stone_bgr,
            1
        )

    help_txt1 = "LMB=add | b=blue | g=green | y=yellow | o=orange"
    help_txt2 = "u=undo | r=reset | s=save | q=save+exit | ESC=exit"
    info_txt = f"Aktuelle Farbe: {current_color}"

    cv2.putText(display, help_txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(display, help_txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.putText(display, help_txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(display, help_txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.putText(display, info_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(display, info_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return display

def build_config(image_reference: str, img_w: int, img_h: int, stones: List[Dict], settings: TeachInSettings) -> Dict:
    return {
        "meta": {
            "image_reference": image_reference,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "plate_size_px": {"w": int(img_w), "h": int(img_h)},
            "plate_size_cm": {"w": float(settings.plate_size_cm[0]), "h": float(settings.plate_size_cm[1])},
            "stone_size_cm": {"w": float(settings.stone_size_cm[0]), "h": float(settings.stone_size_cm[1])},
            "tolerance_norm_default": float(settings.tolerance_norm),
            "allowed_colors": ["blue", "green", "yellow", "orange"]
        },
        "stones": stones
    }

def save_config(cfg: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

def run_teach_in(image_path: str, settings: TeachInSettings) -> Optional[Dict]:
    img, img_w, img_h = load_image(image_path)
    stones: List[Dict] = []
    next_id = 1
    saved_cfg: Optional[Dict] = None
    cfg_path = out_path(settings)

    current_color = "blue"  # Standardfarbe beim Start

    def mouse_callback(event, x, y, flags, param):
        nonlocal next_id, current_color
        if event == cv2.EVENT_LBUTTONDOWN:
            stones.append(make_stone(next_id, x, y, img_w, img_h, settings, current_color))
            print(f"Stein {next_id} gesetzt bei (x={x/img_w:.3f}, y={y/img_h:.3f}, color={current_color})")
            next_id += 1

    cv2.namedWindow(settings.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(settings.window_name, mouse_callback)

    while True:
        cv2.imshow(settings.window_name, draw_overlay(img, stones, img_w, img_h, current_color))
        key = cv2.waitKey(10) & 0xFF

        # Farbauswahl
        if chr(key) in ALLOWED_COLORS:
            current_color = ALLOWED_COLORS[chr(key)][0]
            print(f"🎨 Aktuelle Farbe: {current_color}")

        if key == ord('s'):
            saved_cfg = build_config(image_path, img_w, img_h, stones, settings)
            save_config(saved_cfg, cfg_path)
            print(f"✅ Konfiguration gespeichert: {cfg_path}")

        if key == ord('q'):
            saved_cfg = build_config(image_path, img_w, img_h, stones, settings)
            save_config(saved_cfg, cfg_path)
            print(f"✅ Konfiguration gespeichert: {cfg_path}")
            break

        if key == ord('u'):
            if stones:
                removed = stones.pop()
                next_id = removed["id"]
                print(f"↩️ Entfernt: Stein {removed['id']}")
            else:
                print("ℹ️ Nichts zum Entfernen.")

        if key == ord('r'):
            stones.clear()
            next_id = 1
            print("🔄 Reset: alle Steine gelöscht.")

        if key == 27:
            break

    cv2.destroyAllWindows()
    return saved_cfg
import cv2


def draw_aoi_result(plate_img, config: dict, detected_stones: list, report: dict):
    vis = plate_img.copy()
    h, w = vis.shape[:2]

    # 1) erkannte Steine grün einzeichnen
    for stone in detected_stones:
        x, y, bw, bh = stone["bbox"]
        cx, cy = stone["center_px"]

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)

    # 2) Sollpositionen + Toleranz rot/blau einzeichnen
    default_tol = float(config.get("tolerance_norm_default", 0.03))

    for cfg_stone in config.get("stones", []):
        tx_norm = float(cfg_stone["position_norm"]["x"])
        ty_norm = float(cfg_stone["position_norm"]["y"])

        tol_cfg = cfg_stone.get("tolerance_norm", {})
        tol_x_norm = float(tol_cfg.get("x", default_tol))
        tol_y_norm = float(tol_cfg.get("y", default_tol))

        tx = int(tx_norm * w)
        ty = int(ty_norm * h)

        tol_x_px = int(tol_x_norm * w)
        tol_y_px = int(tol_y_norm * h)

        x1 = tx - tol_x_px
        y1 = ty - tol_y_px
        x2 = tx + tol_x_px
        y2 = ty + tol_y_px

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (tx, ty), 4, (0, 0, 255), -1)

        stone_id = cfg_stone.get("id", "?")
        cv2.putText(
            vis,
            f"id:{stone_id}",
            (tx + 5, ty - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # 3) Status oben links
    status = report.get("status", "UNKNOWN")
    cv2.putText(
        vis,
        f"STATUS: {status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if status == "NIO" else (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return vis
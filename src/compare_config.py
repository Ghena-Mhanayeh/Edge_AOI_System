import math

def compare_stones_to_config(
    detected_stones: list[dict],
    config: dict,
    fail_on_extra: bool = True,
):
    cfg_stones = config.get("stones", [])
    meta = config.get("meta", {})
    default_tol = float(meta.get("tolerance_norm_default", 0.03))

    unused_detected = set(range(len(detected_stones)))
    matches = []
    missing = []
    color_mismatches = []

    for cfg in cfg_stones:
        sid = cfg.get("id", None)
        tx = float(cfg["position_norm"]["x"])
        ty = float(cfg["position_norm"]["y"])
        expected_color = cfg.get("color", None)

        tol = cfg.get("tolerance_norm", {})
        tol_x = float(tol.get("x", default_tol))
        tol_y = float(tol.get("y", default_tol))

        candidates = []
        for i in list(unused_detected):
            dx, dy = detected_stones[i]["center_norm"]

            if abs(dx - tx) <= tol_x and abs(dy - ty) <= tol_y:
                dist = math.hypot(dx - tx, dy - ty)
                candidates.append((dist, i, dx, dy))

        if not candidates:
            missing.append({
                "id": sid,
                "target_norm": (tx, ty),
                "expected_color": expected_color,
                "tolerance_norm": (tol_x, tol_y),
            })
            continue

        candidates.sort(key=lambda t: t[0])
        _, best_i, bx, by = candidates[0]
        unused_detected.remove(best_i)

        detected_color = detected_stones[best_i].get("color", None)
        color_ok = (expected_color == detected_color)

        match_entry = {
            "id": sid,
            "target_norm": (tx, ty),
            "detected_index": best_i,
            "detected_norm": (bx, by),
            "tolerance_norm": (tol_x, tol_y),
            "expected_color": expected_color,
            "detected_color": detected_color,
            "color_ok": color_ok,
        }
        matches.append(match_entry)

        if not color_ok:
            color_mismatches.append({
                "id": sid,
                "detected_index": best_i,
                "target_norm": (tx, ty),
                "detected_norm": (bx, by),
                "expected_color": expected_color,
                "detected_color": detected_color,
            })

    extra = []
    for i in sorted(unused_detected):
        extra.append({
            "detected_index": i,
            "detected_norm": detected_stones[i]["center_norm"],
            "detected_color": detected_stones[i].get("color", None),
            "bbox": detected_stones[i].get("bbox"),
        })

    ok = (
        len(missing) == 0
        and len(color_mismatches) == 0
        and not (fail_on_extra and len(extra) > 0)
    )

    status = "IO" if ok else "NIO"

    report = {
        "status": status,
        "expected_count": len(cfg_stones),
        "detected_count": len(detected_stones),
        "matched_count": len(matches),
        "missing": missing,
        "extra": extra,
        "color_mismatches": color_mismatches,
        "matches": matches,
    }
    return status, report
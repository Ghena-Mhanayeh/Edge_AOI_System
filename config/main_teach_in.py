from config_teach_in import run_teach_in, TeachInSettings

def main():
    settings = TeachInSettings(
        plate_size_cm=(8.0, 6.0),      # Platte 8x6 cm
        stone_size_cm=(0.8, 1.5),      # Stein 0.8x1.5 cm
        tolerance_norm=0.1,           # 10% Toleranz (normiert zur Platte)
        out_dir="config",
        out_name="config.yaml",
        window_name="Teach-In"
    )

    cfg = run_teach_in("config/platte.png", settings)
    if cfg is None:
        print("Beendet ohne Speichern.")
    else:
        print("Teach-In fertig. Anzahl Steine:", len(cfg["stones"]))

if __name__ == "__main__":
    main()
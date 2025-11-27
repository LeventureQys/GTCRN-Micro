import csv
import os

DATASETS_ROOT = "gtcrn_micro/data/dns3_raw"


def make_csv(root_dir, out_csv):
    paths = []
    # getting paths for different datasets
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(r, f))

    paths.sort()
    # writing these files to csvs
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_dir"])
        for p in paths:
            w.writerow([p])


if __name__ == "__main__":
    # getting symlink path
    clean_root = os.path.join(DATASETS_ROOT, "clean")
    noise_root = os.path.join(DATASETS_ROOT, "noise")
    rir_root = os.path.join(DATASETS_ROOT, "impulse_responses")

    # for making training csvs
    make_csv(clean_root, "gtcrn_micro/data/dns3_csvs/train_clean_dir.csv")
    make_csv(noise_root, "gtcrn_micro/data/dns3_csvs/train_noise_dir.csv")
    make_csv(rir_root, "gtcrn_micro/data/dns3_csvs/train_rir_dir.csv")

    print("CSVS created... at ", DATASETS_ROOT)

import argparse
import os
import subprocess
import shutil
import zipfile
import pandas as pd

# 1) Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, required=True, help="where to save the dataset")
args = parser.parse_args()

# 2) Prepare folders
subdirs = [
    "csv",
    "clips_videos/train", "clips_videos/val",
    "clips_audios/train", "clips_audios/val"
]
for sub in subdirs:
    os.makedirs(os.path.join(args.save_path, sub), exist_ok=True)

# 3) Utility to download & unzip
def download_and_extract(blob_url: str, output_path: str):
    raw_url = blob_url.replace("/blob/", "/resolve/")
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    # clean prior
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    elif os.path.exists(output_path):
        os.remove(output_path)
    # download
    subprocess.run(["wget", raw_url, "-O", output_path], check=True)
    # unzip & delete zip
    with zipfile.ZipFile(output_path, 'r') as z:
        z.extractall(path=out_dir)
    os.remove(output_path)

# 4) Column names (no header in per-video CSVs)
columns = [
    "video_id", "frame_timestamp",
    "entity_box_x1", "entity_box_y1",
    "entity_box_x2", "entity_box_y2",
    "label", "entity_id",
    "label_id", "instance_id"
]

# 5) Process each split separately
for split in ["train", "val"]:
    df_list = []

    with open(os.path.join("video_list", f"{split}.csv"), "r") as f:
        for line in f:
            if line.strip() == "Link":
                continue
            video_name = line.strip().split("v=")[-1]

            # a) Video ZIP
            vid_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"clips_videos/{split}/{video_name}.zip"
            )
            out_vid = os.path.join(args.save_path, "clips_videos", split, f"{video_name}.zip")
            download_and_extract(vid_url, out_vid)

            # b) Audio ZIP
            aud_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"clips_audios/{split}/{video_name}.zip"
            )
            out_aud = os.path.join(args.save_path, "clips_audios", split, f"{video_name}.zip")
            download_and_extract(aud_url, out_aud)

            # c) CSV download + read
            csv_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"csv/{split}/{video_name}.csv"
            )
            out_csv = os.path.join(args.save_path, "csv", f"{split}_{video_name}.csv")
            raw_csv = csv_url.replace("/blob/", "/resolve/")
            subprocess.run(["wget", raw_csv, "-O", out_csv], check=True)

            # push to list before writing into a big file
            df = pd.read_csv(out_csv, header=None, names=columns)
            df_list.append(df)

            os.remove(out_csv)

    # Merge & write {split}_orig.csv
    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        out_merged = os.path.join(args.save_path, "csv", f"{split}_orig.csv")
        merged.to_csv(out_merged, index=False)
        print(f"Wrote merged CSV for {split}: {out_merged}")

    break
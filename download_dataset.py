import argparse
import os
import cmd

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type = str, required=True, help = "path to save the dataset")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, "csv"), exist_ok=True)
os.makedirs(os.path.join(args.save_path, "clips_videos"), exist_ok=True)
os.makedirs(os.path.join(args.save_path, "clips_audios"), exist_ok=True)

for tt in ["train", "val"]:
    with open(os.path.join("video_list", f"{tt}.csv"), "r") as f:
        for u in f:
            if u.strip() == "Link":
                continue
            video_name = u.strip().split("v=")[-1]
            print(video_name)

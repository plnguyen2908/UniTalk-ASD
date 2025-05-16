import os
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, help="source dataset")
parser.add_argument("--destination", type=str, help="target dataset")
args = parser.parse_args()

os.makedirs(args.destination, exist_ok=True)
shutil.rmtree(args.destination)

print("begin copying")
shutil.copytree(os.path.join(args.source, "csv"), os.path.join(args.destination, "csv"))
os.makedirs(os.path.join(args.destination, "clips_audios", "train"), exist_ok=True)
os.makedirs(os.path.join(args.destination, "clips_audios", "val"), exist_ok=True)
os.makedirs(os.path.join(args.destination, "clips_videos", "train"), exist_ok=True)
os.makedirs(os.path.join(args.destination, "clips_videos", "val"), exist_ok=True)
t = ["val"]

print("begin moving")
for tt in t:
    for video in os.listdir(os.path.join(args.source, "clips_videos", tt)):
        for entity in os.listdir(os.path.join(args.source, "clips_videos", tt, video)):
            shutil.copytree(os.path.join(args.source, "clips_videos", tt, video, entity), os.path.join(args.destination, "clips_videos", tt, entity))
    
for tt in t:
    for video in os.listdir(os.path.join(args.source, "clips_audios", tt)):
        for entity in os.listdir(os.path.join(args.source, "clips_audios", tt, video)):
            shutil.copy(os.path.join(args.source, "clips_audios", tt, video, entity), os.path.join(args.destination, "clips_audios", tt, entity))

df = pd.read_csv(os.path.join(args.source, "csv", "val_orig.csv"))

os.makedirs(os.path.join(args.destination, "gt"), exist_ok=True)
for vid, group in df.groupby('video_id'):
    fname = os.path.join(args.destination, "gt", f"{vid}.csv")
    group.to_csv(fname, index=False, header=False)

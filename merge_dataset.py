import argparse
import os
import shutil
import pandas as pd
import tqdm

print("begin")

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--add", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

if os.path.exists(args.output):
    shutil.rmtree(args.output)

shutil.copytree(args.base, args.output)

df_base = pd.read_csv(f"{args.base}/csv/train_orig.csv")
df_add = pd.read_csv(f"{args.add}/csv/train_orig.csv")

unique_ids = df_base['video_id'].unique()


for (idx, vid_name) in tqdm.tqdm(enumerate(os.listdir(f"{args.add}/clips_audios/train"))):
    assert vid_name not in unique_ids
    shutil.copytree(f"{args.add}/clips_audios/train/{vid_name}", f"{args.output}/clips_audios/train/{vid_name}")
    shutil.copytree(f"{args.add}/clips_videos/train/{vid_name}", f"{args.output}/clips_videos/train/{vid_name}")

df_res = pd.concat([df_base, df_add], axis=0, ignore_index=True)
df_res.to_csv(f"{args.output}/csv/train_orig.csv")

with open(f"{args.output}/csv/train_loader.csv", 'a') as f:
    with open(f"{args.add}/csv/train_loader.csv", "r") as g:
        for row in g:
            f.writelines(row)


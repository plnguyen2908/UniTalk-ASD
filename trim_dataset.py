import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="path of the dataset you want to trim")
parser.add_argument("--new_path", type=str, help="destination path you want to copy to")
parser.add_argument("--list", type=str, help="path to the csv file containing the list of videos id to keep")
args = parser.parse_args()

path = args.path
new_path = args.new_path

import pandas as pd
import os
import shutil
import numpy as np

df = pd.read_csv(f'{path}/csv/val_orig.csv')


# unique_ids = df['video_id'].unique()

# # Randomly sample half of the unique video_ids.
# half_ids = np.random.choice(unique_ids, size=len(unique_ids) // 2, replace=False)
half_ids = []
with open(args.list, "r") as f:
    for id in f:
        half_ids.append(id.strip())

# Filter the DataFrame to only include rows with the sampled video_ids.
df_half = df[df['video_id'].isin(half_ids)]


if os.path.exists(new_path):
    shutil.rmtree(new_path)

os.makedirs(new_path, exist_ok = True)
os.makedirs(f"{new_path}/clips_videos/val")
os.makedirs(f"{new_path}/clips_audios/val")

os.makedirs(f"{new_path}/csv")

for vid_name in os.listdir(f"{path}/clips_audios/val"):
    if vid_name in half_ids:
        shutil.copytree(f"{path}/clips_audios/val/{vid_name}", f"{new_path}/clips_audios/val/{vid_name}")
        shutil.copytree(f"{path}/clips_videos/val/{vid_name}", f"{new_path}/clips_videos/val/{vid_name}")

df_half.to_csv(f"{new_path}/csv/val_orig.csv")


with open(f"{new_path}/csv/val_loader.csv", 'w') as f:
    with open(f"{path}/csv/val_loader.csv", "r") as g:
        for row in g:
            l = row.split('\t')
            vid_name = l[0][:11]
            if vid_name in half_ids:
                f.write(row)
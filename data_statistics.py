import os
import pandas as pd
import argparse
from deepface import DeepFace
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="path to the dataset you want to make statistics")
args = parser.parse_args()

train_orig_path = os.path.join(args.path, "csv", "train_orig.csv")
val_orig_path = os.path.join(args.path, "csv", "val_orig.csv")

train_orig = pd.read_csv(train_orig_path)
val_orig = pd.read_csv(val_orig_path)

train_total_rows = len(train_orig)
val_total_rows = len(val_orig)

# 2. Group by `entity_id` and count how many times each appears
train_num_groups = train_orig.groupby("entity_id").ngroups
val_num_groups = val_orig.groupby("entity_id").ngroups

# since all of the videos are sampled in 25fps, divide by 25 to see the number of total seconds
# for the dataset


train_counts = train_orig.groupby('label_id').size()
val_counts = val_orig.groupby('label_id').size()
ones_count = train_counts.loc[1] + val_counts.loc[1]

tt = ["train", "val"]
cnt = {}
gender = {}
race = {}
ss = {"small": 0, "medium": 0, "large": 0}

for t in tt:
    for video in os.listdir(os.path.join(args.path, "clips_videos", t)):
        time_track = {}
        for entity in os.listdir(os.path.join(args.path, "clips_videos", t, video)):
            for face in os.listdir(os.path.join(args.path, "clips_videos", t, video, entity)):
                if face[:-4] not in time_track:
                    time_track[face[:-4]] = 0
                time_track[face[:-4]] += 1
                objs = DeepFace.analyze(
                    img_path = os.path.join(args.path, "clips_videos", t, video, entity, face),
                    detector_backend = "skip", 
                    actions = ['gender', 'race']
                )
                objs = objs[0]
                if objs["dominant_gender"] not in gender:
                    gender[objs["dominant_gender"]] = 0
                gender[objs["dominant_gender"]] += 1
                if objs["dominant_race"] not in race:
                    race[objs["dominant_race"]] = 0
                race[objs["dominant_race"]] += 1
                face_img = cv2.imread(os.path.join(args.path, "clips_videos", t, video, entity, face))
                res =  face_img.shape[0] * face_img.shape[1]
                if res < 64 * 64:
                    ss["small"] += 1
                elif res < 128 * 128:
                    ss["medium"] += 1
                else:
                    ss["large"] += 1

#             # break
        for key in time_track:
            if time_track[key] not in cnt:
                cnt[time_track[key]] = 0
            cnt[time_track[key]] += 1
        # break
    # break

print(f"Total number of faces: {train_total_rows + val_total_rows}")
print(f"Total number of face tracks: {train_num_groups + val_num_groups}")
print(f"Total hours of the dataset: {(train_total_rows + val_total_rows) / (25 * 3600)}")
print(f"Speech time: {ones_count / (25 * 3600)}")
print(f"Statistics for number of faces per frame: {cnt}")
print(f"Statistics for gender: {gender}")
print(f"Statistics for race: {race}")
print(f"Statistics for face size: {race}")

frames = 0
for key in cnt:
    frames += cnt[key]
print(f"Avg faces per frame: {(train_total_rows + val_total_rows) / frames}")


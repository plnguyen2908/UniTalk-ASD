# UniTalk-ASD

[Le Thien Phuc Nguyen*](https://plnguyen2908.github.io/), [Zhuoran Yu*](https://www.zhuoranyu.com/), Khoa Cao Quang Nhat, Yuwei Guo, Tu Ho Manh Pham, Tuan Tai Nguyen, Toan Ngo Duc Vo, Lucas Poon, [Soochahn Lee](https://sites.google.com/view/soochahnlee/),  [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (* *Equal Contribution*)

## Downloading the dataset:

To download the dataset, you just need to excute the following file (installation requirement: python, pandas, and wget)

```
python download_dataset.py --save_path /path/to/the/dataset
```

where --save_path is the path to the folder where you want to store the dataset (that path doesn't have to exist, we will create one for you). For your information, it takes 834.38 seconds to download everything on a macbook air m4. After downloading, you will get the following dataset's structure:

```
root/
├── csv/
│   ├── val_orig.csv
│   └── train_orig.csv
├── clips_audios/
│   ├── train/
│   │   └── <video_id>/
│   │       └── <entity_id>.wav
│   └── val/
│       └── <video_id>/
│           └── <entity_id>.wav
└── clips_videos/
    ├── train/
    │   └── <video_id>/
    │       └── <entity_id>/
    │           ├── <time>.jpg (face)
    │           └── <time>.jpg (face)
    └── val/
        └── <video_id>/
            └── <entity_id>/
                ├── <time>.jpg (face)
                └── <time>.jpg (face)
```

## Exploring the dataset
- Inside the csv folder, there are 2 csv files for training and testing. In each csv files, each row represents a face, and there are 10 columns where:
  - **video_id**: the id of the video
  - **frame_timestamp**: the timestamp of the face in video_id
  - **entity_box_x1**, **entity_box_y1**, **entity_box_x2**, **entity_box_y2**: the relative coordinate of the bounding box of the face
  - **label**: SPEAKING_AUDIBLE or NOT_SPEAKING
  - **entity_id**: the id of the face tracks (a set of consecutive faces of the same person) in the format video_id:number
  - **label_id**: 1 or 0
  - **instance_id**: consecutive faces of an entity_id which are always not speaking are speaking. It is in the format entity_id:number
- Inside clips_audios, there are 2 folders which are train and val splits. In each split, there will be a list of video_id folder which contains the audio file (in form of wav) for each entity_id.
- Inside clips_videos, there are 2 folders which are train and val splits. In each split, there will be a list of video_id folder in which each contains a list of entity_id folder. In each entity_id folder, there are images of the face of that entity_id person.
- We sample the video at 25 fps. So, if you want to use other cues to support the face prediction, we would recommend checking the video_list folder which contains the link to the list of videos we use. You can download it and sample at 25 fps.

## Evaluation
- We follow [AVA-ActiveSpeaker](https://arxiv.org/abs/1901.01342) by using the mAP metric. The python to run mAP metric is in the tool folder. You just need to run:
```
python tool/get_ava_active_speaker_performance.py -g groundtruth.csv -p prediction.csv
```

Little notice, if you read the file, it requires you to put the label to SPEAKING_AUDIBLE in each row because as mentioned in our paper, the metric is calculated on the score sorted decreasingly.

## Sub-categories evaluation dataset
- To create the evaluation dataset of sub-category, you can run the following command:
```
python trim_dataset.py --path /path/to/your/dataset --new_path /path/to/store/your/new/evaluation/set --list sub_categories/test_(category).csv
```

The trim_dataset.py will take the subset of the test set of your dataset and store it into a new path. It use the sub_categories/test_<category>.csv as guidance to take which videos.

## Convert to ASC and ASDNet's dataset structure
Since ASC and ASDNet use different dataset's structure, you can use the file convert_to_ASC.py by running:

```
python convert_to_ASC.py --source /path/to/your/data --destination /path/to/new/dataset
```

## Loading each entity's id information from Huggging Face

We also provide a way to load the information of each entity_id (i.e, face track) through the hub of huggingface. However, this method is less flexible and cannot be used for models that use multiple face tracks like ASDNet or LoCoNet. You just need to run:

```
from datasets import load_dataset
dataset = load_dataset("plnguyen2908/UniTalk", split = "train|val", trust_remote_code=True)
```

This method is more memory-efficient. However, its drawback is speed (around 20-40 hours to read all instances of face tracks) and less flexible than the first method.

For each instance, it will return:

```
{
    "entity_id": the id of the face track
    "images": list of images of face crops of the face_track
    "audio": the audio that has been read from wavfile.read
    "frame_timestamp": time of each face crop in the video
    "label_id": the label of each face (0 or 1)
}
```

## Pretrained weights:

### Top performing models:

| Model           | Weight                                                                                        | Trained on        |
| --------------- | --------------------------------------------------------------------------------------------- | ----------------- |
| TalkNCE | [Checkpoint](https://drive.google.com/file/d/1eBc5xn7I32__cNPupYYx24AdwksUs-8F/view?usp=sharing) | UniTalk |
| LoCoNet | [Checkpoint](https://drive.google.com/file/d/10HvRqNO34QsJpZgQTMYZdIAW4A2kP5fw/view?usp=sharing) | UniTalk |
| TalkNet | [Checkpoint](https://drive.google.com/file/d/1Uq8vp__7UywtdY0z5zz8QtcbEwFfXlNw/view?usp=sharing) | UniTalk |

### Fined-tune UniTalk-pretrained TalkNCE on AVA:

| Model           | Weight                                                                                        | 
| --------------- | --------------------------------------------------------------------------------------------- |
| 3h | [Checkpoint](https://drive.google.com/file/d/1z0hEK-QIPyXeBmatRSA6gyLiVM-Hur3_/view?usp=sharing) | 
| 5h | [Checkpoint](https://drive.google.com/file/d/1FkUL625DoxwtFMXHQC2z_9YZCWZBK2vz/view?usp=sharing) | 
| 10h | [Checkpoint](https://drive.google.com/file/d/1-yEjMVk_xztsYveLuLoA4kWfXR9JYyKW/view?usp=sharing) | 
| 15h | [Checkpoint](https://drive.google.com/file/d/1riiHL3skpjKl1nk4OGnedz-ov-BKIHDt/view?usp=sharing) |
| full AVA | [Checkpoint](https://drive.google.com/file/d/1BvRhVhyZwUv9-bsPskARTdMEPk5XyRYg/view?usp=sharing)|


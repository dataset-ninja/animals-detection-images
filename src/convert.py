# Path to the original dataset

import os

import gdown
import numpy as np
import supervisely as sly
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)

import src.settings as s

dataset_path = "./APP_DATA/archive"
# items_folder = "data"
batch_size = 30
images_ext = ".jpg"
anns_ext = ".txt"


def create_ann(image_path, ds_name, cls_name, cls_to_obj_classes):
    labels = []

    image_np = sly.imaging.image.read(image_path)[:, :, 0]
    img_height = image_np.shape[0]
    img_width = image_np.shape[1]

    bbox_path = os.path.join(
        dataset_path, ds_name, cls_name, "Label", get_file_name(image_path) + anns_ext
    )

    if file_exists(bbox_path):
        with open(bbox_path) as f:
            content = f.read().split("\n")

            for curr_data in content:
                if len(curr_data) != 0:
                    line = curr_data.split(" ")
                    obj_class = cls_to_obj_classes[line.pop(0)]

                    curr_data = list(map(float, line))

                    top = curr_data[1]
                    left = curr_data[0]
                    right = curr_data[2]
                    bottom = curr_data[3]

                    rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                    label = sly.Label(rectangle, obj_class)
                    labels.append(label)

    return sly.Annotation(img_size=(img_height, img_width), labels=labels)


def get_img_basenames(folder_path):
    img_basenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(images_ext):
                img_basenames.append(os.path.basename(file).split(".")[0])
    return img_basenames


class_names = [
    "Bear",
    "Brown bear",
    "Bull",
    "Butterfly",
    "Camel",
    "Canary",
    "Caterpillar",
    "Cattle",
    "Centipede",
    "Cheetah",
    "Chicken",
    "Crab",
    "Crocodile",
    "Deer",
    "Duck",
    "Eagle",
    "Elephant",
    "Fish",
    "Fox",
    "Frog",
    "Giraffe",
    "Goat",
    "Goldfish",
    "Goose",
    "Hamster",
    "Harbor seal",
    "Hedgehog",
    "Hippopotamus",
    "Horse",
    "Jaguar",
    "Jellyfish",
    "Kangaroo",
    "Koala",
    "Ladybug",
    "Leopard",
    "Lion",
    "Lizard",
    "Lynx",
    "Magpie",
    "Monkey",
    "Moths and butterflies",
    "Mouse",
    "Mule",
    "Ostrich",
    "Otter",
    "Owl",
    "Panda",
    "Parrot",
    "Penguin",
    "Pig",
    "Polar bear",
    "Rabbit",
    "Raccoon",
    "Raven",
    "Red panda",
    "Rhinoceros",
    "Scorpion",
    "Sea lion",
    "Sea turtle",
    "Seahorse",
    "Shark",
    "Sheep",
    "Shrimp",
    "Snail",
    "Snake",
    "Sparrow",
    "Spider",
    "Squid",
    "Squirrel",
    "Starfish",
    "Swan",
    "Tick",
    "Tiger",
    "Tortoise",
    "Turkey",
    "Turtle",
    "Whale",
    "Woodpecker",
    "Worm",
    "Zebra",
]


obj_classes = []
cls_to_obj_classes = {}
for cls in class_names:
    obj_classes.append(sly.ObjClass(cls, sly.Rectangle))
    cls_to_obj_classes[cls] = sly.ObjClass(cls, sly.Rectangle)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=obj_classes)
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in ["train", "test"]:
        images_names = get_img_basenames(os.path.join(dataset_path, ds_name))

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for cls_name in os.listdir(os.path.join(dataset_path, ds_name)):
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [
                    os.path.join(dataset_path, ds_name, cls_name, im_name + images_ext)
                    for im_name in img_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [
                    create_ann(image_path, ds_name, cls_name, cls_to_obj_classes)
                    for image_path in img_pathes_batch
                ]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))

    return project

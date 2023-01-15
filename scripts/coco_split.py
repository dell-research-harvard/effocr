import numpy as np
import json
import argparse
import os
import copy
import random


def split_coco_json(coco_json, tvt_split, seed=99):

    train_pct, val_pct, test_pct = tvt_split
    assert 0.9999 < train_pct + val_pct + test_pct <= 1

    images = coco_json["images"]
    annotations = coco_json["annotations"]
    image_ids = [im["id"] for im in images]

    train_N = int(train_pct * len(images))
    val_N = int(val_pct * len(images))

    random.seed(seed)
    image_ids = random.sample(image_ids, k=len(image_ids))

    train_image_ids = image_ids[:train_N]
    val_image_ids = image_ids[train_N:train_N+val_N]
    test_image_ids = image_ids[train_N+val_N:]

    train_images = [im for im in images if im["id"] in train_image_ids]
    val_images = [im for im in images if im["id"] in val_image_ids]
    test_images = [im for im in images if im["id"] in test_image_ids]

    train_annotations = [a for a in annotations if a["image_id"] in train_image_ids]
    val_annotations = [a for a in annotations if a["image_id"] in val_image_ids]
    test_annotations = [a for a in annotations if a["image_id"] in test_image_ids]

    assert len(train_images) + len(val_images) + len(test_images) == len(images)
    assert len(set(train_image_ids).intersection(set(val_image_ids))) == 0
    assert len(set(val_image_ids).intersection(set(test_image_ids))) == 0
    assert len(set(test_image_ids).intersection(set(train_image_ids))) == 0
    assert len(train_annotations) + len(val_annotations) + len(test_annotations) == len(annotations)

    return train_images, val_images, test_images, train_annotations, val_annotations, test_annotations
    

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True,
        help="Absolute path to COCO JSON of interest")
    parser.add_argument("--tvt_split", type=str, required=True,
        help="Train/validation/test split proportions, comma-separated")
    args = parser.parse_args()

    with open(args.coco_json) as f:
        coco_json = json.load(f)
    train_coco_json = copy.deepcopy(coco_json)
    val_coco_json = copy.deepcopy(coco_json)
    test_coco_json = copy.deepcopy(coco_json)

    tvt_split = [float(x) for x in args.tvt_split.split(",")]
    train_images, val_images, test_images, \
        train_annotations, val_annotations, test_annotations = \
            split_coco_json(coco_json, tvt_split)

    train_coco_json["images"] = train_images
    train_coco_json["annotations"] = train_annotations

    val_coco_json["images"] = val_images
    val_coco_json["annotations"] = val_annotations

    test_coco_json["images"] = test_images
    test_coco_json["annotations"] = test_annotations

    input_basename, _ = os.path.splitext(os.path.basename(args.coco_json))
    src_dir = os.path.dirname(args.coco_json)
    pct_train, pct_val, pct_test = [int(100*x) for x in tvt_split]

    with open(os.path.join(src_dir, input_basename.replace("all", f"train{pct_train}") + ".json"), "w") as f:
        json.dump(train_coco_json, f, indent=2)
    with open(os.path.join(src_dir, input_basename.replace("all", f"val{pct_val}") + ".json"), "w") as f:
        json.dump(val_coco_json, f, indent=2)
    with open(os.path.join(src_dir, input_basename.replace("all", f"test{pct_test}") + ".json"), "w") as f:
        json.dump(test_coco_json, f, indent=2)

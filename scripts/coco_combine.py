import numpy as np
import json
import argparse
import subprocess
import os
import shutil
from tqdm import tqdm


def process_coco_json(coco_json, input_n, tag):
    images = coco_json["images"]
    subset_images = images if input_n is None else np.random.choice(images, input_n, replace=False) 
    subset_image_ids = [im["id"] for im in subset_images]
    annotations = coco_json["annotations"]
    subset_annotations = [a for a in annotations if a["image_id"] in subset_image_ids]
    tagged_subset_images = subset_images.copy()
    tagged_subset_annotations = subset_annotations.copy()
    _ = [im.update({"id": str(im["id"]) + tag}) for im in tagged_subset_images]
    _ = [a.update({"image_id": str(a["image_id"]) + tag}) for a in tagged_subset_annotations]
    _ = [a.update({"id": str(a["id"]) + tag}) for a in tagged_subset_annotations]
    return list(tagged_subset_images), list(tagged_subset_annotations)


def combine_coco_images_and_annotations(images_1, images_2, annotations_1, annotations_2):
    combined_images = images_1 + images_2
    combined_annotations = annotations_1 + annotations_2
    new_image_id = 0
    for im in tqdm(combined_images):
        old_image_id = im["id"]
        im["id"] = new_image_id
        for a in combined_annotations:
            if a["image_id"] == old_image_id:
                a["image_id"] = new_image_id
        new_image_id += 1
    new_anno_id = 0
    for a in tqdm(combined_annotations):
        a["id"] = new_anno_id
        new_anno_id += 1
    return combined_images, combined_annotations
    

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", type=str, required=True,
        help="Absolute path to one COCO JSON to be merged")
    parser.add_argument("--json2", type=str, required=True,
        help="Absolute path to another COCO JSON to be merged")
    parser.add_argument("--imdir1", type=str, required=True,
        help="Absolute path to one COCO JSON relevant image directory to be merged")
    parser.add_argument("--imdir2", type=str, required=True,
        help="Absolute path to another COCO JSON relevant image directory to be merged")
    parser.add_argument("--outjsonname", type=str, required=True,
        help="Name of combined COCO JSON file")
    parser.add_argument("--outdir", type=str, required=True,
        help="Absoluate path to output directory")
    args = parser.parse_args()

    with open(args.json1) as f:
        coco_json_1 = json.load(f)
        images_1, annotations_1 = process_coco_json(coco_json_1, None, tag="inp1")
        categories_1 = {c["id"]:c["name"] for c in coco_json_1["categories"]}

    with open(args.json2) as f:
        coco_json_2 = json.load(f)
        images_2, annotations_2 = process_coco_json(coco_json_2, None, tag="inp2")
        categories_2 = {c["id"]:c["name"] for c in coco_json_2["categories"]}

    print("Combining JSONs!")
    combo_images, combo_annotations = \
        combine_coco_images_and_annotations(images_1, images_2, annotations_1, annotations_2)

    combo_coco_json = {
        "images": combo_images,
        "annotations": combo_annotations,
        "info": {"year": 2022, "version": "1.0", "contributor": "synth-textlines"},
        "categories": [{"id": idx, "name": cat} for idx, cat in \
            enumerate(set(list(categories_1.values()) + list(categories_2.values())))],
        "licenses": ""
    }

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "images"), exist_ok=True)

    with open(os.path.join(args.outdir, args.outjsonname), "w") as f:
        json.dump(combo_coco_json, f, indent=2)

    print("Combining image directories!")
    for im in tqdm(combo_images):
        path1 = os.path.join(args.imdir1, im["file_name"])
        path2 = os.path.join(args.imdir2, im["file_name"])
        if os.path.isfile(path1) and os.path.isfile(path2):
            raise NameError
        path = path1 if os.path.isfile(path1) else path2
        assert os.path.isfile(path)
        if os.path.isfile(os.path.join(args.outdir, "images", os.path.basename(path))):
            continue
        shutil.copy(path, os.path.join(args.outdir, "images"))

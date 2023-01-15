import numpy as np
import json
import argparse
import os
import copy


def subset_coco_json(coco_json, N):
    images = coco_json["images"]
    subset_images = np.random.choice(images, N, replace=False).tolist()
    subset_image_ids = [im["id"] for im in subset_images]
    annotations = coco_json["annotations"]
    subset_annotations = [a for a in annotations if a["image_id"] in subset_image_ids]
    return subset_images, subset_annotations
    

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_coco_json", type=str, required=True,
        help="Absolute path of COCO JSON to be subset")
    parser.add_argument("--n", type=int, required=True,
        help="Subset of size N")
    args = parser.parse_args()

    with open(args.input_coco_json) as f:
        coco_json = json.load(f)
    sub_coco_json = copy.deepcopy(coco_json)
    sub_images, sub_annotations = subset_coco_json(coco_json, args.n)
    sub_coco_json["images"] = sub_images
    sub_coco_json["annotations"] = sub_annotations

    with open(os.path.join(os.path.dirname(args.input_coco_json), 
            f"{os.path.splitext(os.path.basename(args.input_coco_json))[0]}_subset{args.n}.json"), "w") as f:
        json.dump(sub_coco_json, f, indent=2)

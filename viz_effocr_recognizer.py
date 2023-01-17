from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm
from torchvision import transforms as T
import faiss
import os

from models.encoders import *
from models.classifiers import *
from effocr_datasets.recognizer_datasets import *
from utils.datasets_utils import *


def create_subtitle(pil, txt, font):

    w, h = pil.size
    background = Image.new('RGB', (w, h+30), (255,255,255))
    draw = ImageDraw.Draw(background)

    if txt != "NN 0":
        draw.text((0,h), txt, (0,0,0), font=font)

    background.paste(pil, (0,0))
    return background


def infer_viz(query_paths, ref_dataset, model, index_path, transform, inf_save_path, local_font, n=100, k=10, seed=111, query_txt=False):

    # inverse image normalization for viz function

    inv_normalize = T.Normalize(
        mean= [-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std= [1/s for s in [0.229, 0.224, 0.225]]
    )

    # load recognizer for inference

    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
    infm = InferenceModel(model, knn_func=knn_func)
    infm.load_knn_func(index_path)

    # optionally perform inference on text list of paths to images

    if query_txt:
        with open(query_paths) as f:
            query_paths = f.read().split()

    # randomness

    np.random.seed(seed)
    np.random.shuffle(query_paths)

    # set viz font

    font = ImageFont.truetype(local_font, 30)
    
    # create visualizations!

    if n is None: n = len(query_paths)
    counter = 0

    for idx, query_path in tqdm(enumerate(query_paths)):

        if counter == n:
            break

        else:

            # get image
            query_stem, _ = os.path.splitext(os.path.basename(query_path))
            query_img = Image.open(query_path).convert("RGB")
            query = transform(query_img).unsqueeze(0)

            # inference
            _, indices = infm.get_nearest_neighbors(query, k=k)

            # get NN images, labels
            nearest_imgs = [inv_normalize(query).squeeze(0)] + [inv_normalize(ref_dataset[i][0]) for i in indices[0]]
            nearest_pils = [T.Resize((80,80))(MedianPad(override=(255,255,255))(T.ToPILImage()(x))) for x in nearest_imgs]
            nearest_subbed = [T.ToTensor()(create_subtitle(x, f"NN {idx}", font)) for idx, x in enumerate(nearest_pils)]
            nearest_chars = [os.path.basename(ref_dataset.data[i][0]).split("_")[0] for i in indices[0]]
            nearest_chars_hex = [hex(ord(c)) if not c.startswith("0x") else c for c in nearest_chars]

            # determine correctness of labels (True/False designation of whether or not 1-NN is correct in filename)
            corr_label = '_'.join([f"nn{idx+1},{x}" for idx, x in enumerate(nearest_chars_hex)])
            nn_matcher = query_stem.split('_')[-2] if "PAIRED" in query_stem else query_stem[-1]
            matched = nn_matcher==nearest_chars_hex[0]

            # save
            torchvision.utils.save_image(
                torchvision.utils.make_grid(nearest_subbed), 
                os.path.join(inf_save_path, 
                    f"q_knn_{idx}_{query_stem}_{matched}_{corr_label}.png")
                )
            
            counter += 1


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir_path", type=str, default="./ref_dir",
        help="Root image directory path, with character class subfolders")
    parser.add_argument("--anno_path", type=str,
        help="Path to train or test or val annotations")
    parser.add_argument("--run_name", type=str, required=True,
        help="Name of W&B run/dir with checkpoints of interest")
    parser.add_argument('--N_to_ocr', type=int, default=None,
        help="Number of random images to OCR")
    parser.add_argument('--k_nn', type=int, default=10,
        help="Number of nearest neighbors")
    parser.add_argument('--query_text', action='store_true', default=False,
        help="Inference on images declared via paths in a text file")
    parser.add_argument('--diff_sizes', action='store_true', default=False,
        help="DEPRECATED: process char crops of different sizes, without resizing them")
    parser.add_argument('--ad_hoc', type=str,
        help="Visualize a single image, ad hoc")
    parser.add_argument("--lang", type=str, default="jp",
        help="Language of characters being recognized")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument('--N_classes', type=int, default=None,
        help="Triggers use of FFNN classifier head with N classes")
    parser.add_argument("--local_font", type=str, default="/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        help="Local font to be used for viz outputs")
    args = parser.parse_args()

    # setup

    RESULTS_DIR = "viz"
    os.makedirs(os.path.join(args.run_name, RESULTS_DIR), exist_ok=True)
    
    # load encoder

    if args.auto_model_hf is None and args.auto_model_timm is None:
        raise NotImplementedError
    elif not args.auto_model_timm is None:
        encoder = AutoEncoderFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None:
        encoder = AutoEncoderFactory("hf", args.auto_model_hf)
    elif not args.auto_model_timm is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("hf", args.auto_model_hf)

    # load best checkpoint

    enc = encoder.load(os.path.join(args.run_name, "enc_best.pth"))

    # get transform

    transform = create_paired_transform()

    # construct separate datasets for paired and rendered images

    paired_dataset = create_paired_dataset(args.ref_dir_path, args.lang)
    render_dataset = create_render_dataset(args.ref_dir_path, args.lang, 
        font_name="NotoSerifCJKjp-Regular" if args.lang == "jp" else "NotoSerif-Regular")

    # ad hoc

    if args.ad_hoc:
        infer_viz(
            query_paths=args.ad_hoc.split(","), 
            ref_dataset=render_dataset, 
            model=enc, 
            index_path=os.path.join(args.run_name, "ref.index"), 
            transform=transform, 
            inf_save_path=os.path.join(args.run_name, RESULTS_DIR), 
            n=args.N_to_ocr, 
            k=args.k_nn, 
            query_txt=args.query_text,
            local_font=args.local_font
        )
        exit(0)
    
    # knn inference

    with open(args.anno_path) as f:
        coco_anno = json.load(f)
        seg_ids = [os.path.splitext(x['file_name'])[0] for x in coco_anno['images']]
        query_paths = [x[0] for x in paired_dataset.data if any(f"PAIRED_{y}_" in x[0] for y in seg_ids)]
        print(f"Num query paths = {len(query_paths)}")

    infer_viz(
        query_paths=query_paths, 
        ref_dataset=render_dataset, 
        model=enc, 
        index_path=os.path.join(args.run_name, "ref.index"), 
        transform=transform, 
        inf_save_path=os.path.join(args.run_name, RESULTS_DIR), 
        n=args.N_to_ocr, 
        k=args.k_nn, 
        query_txt=args.query_text
    )

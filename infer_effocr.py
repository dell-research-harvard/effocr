import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import faiss
from tqdm import tqdm
import json
import argparse
import numpy as np
from glob import glob
import os
import sys
import io
import requests
import base64
import copy
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from mmdet.apis import init_detector, inference_detector
import mmcv
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/issues/59
sys.path.insert(0, "../")

from utils.datasets_utils import *
from models.encoders import *
from datasets.effocr_datasets import *
from utils.coco_utils import *
from utils.eval_utils import *
from utils.spell_check_utils import *
from models.classifiers import *
from datasets.recognizer_datasets import create_render_dataset


def run_gcv(
        image_file, 
        client, 
        lang="ja"
    ):
    """Call to GCV OCR"""

    with io.open(image_file, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image, image_context={"language_hints": [lang]})
    document = response.full_text_annotation
    return document.text


def run_baidu(
        image_path,
        access_token,
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
        lang="JAP"
    ):
    """Call to Baidu OCR"""

    with open(image_path, 'rb') as f:
        img = base64.b64encode(f.read())
    params = {"image":img,"language_type":lang}
    request_url = f"{request_url}?access_token={access_token}"
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    response_json = response.json()
    if response:
        return "".join(x['words'] for x in response_json['words_result'])
    else:
        print("Baidu OCR call returned nothing...")
        return None


def create_dataset(image_paths, transform):
    """Create dataset for inference"""

    dataset = EffOCRInferenceDataset(image_paths, transform=transform)
    print(f"Length inference dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    return dataloader


def gt_collect(results, gts):
    gt_pred_pairs = []
    for fn, gt in gts:
        pred = results.get(fn, None)
        if pred is None:
            gt_pred_pairs.append((gt, ""))
        else:
            gt_pred_pairs.append((gt, pred))
    return gt_pred_pairs


class EffOCR:

    def __init__(self,
            localizer_checkpoint,
            localizer_config,
            recognizer_checkpoint,
            recognizer_index,
            recognizer_chars,
            class_map,
            encoder,
            image_dir,
            vertical,
            char_transform,
            lang,
            device,
            save_chars=True,
            blacklist=None,
            score_thresh=0.5,
            score_thresh_word=0.5,
            knn=10,
            spell_check=False,
            N_classes=None,
            anchor_margin=None,
            d2=False,
            ad_hoc_index_root_dir=None
        ):
        # load localizer

        if not d2:
            if lang == "en":
                loc_config = {
                    "model.rpn_head.anchor_generator.scales":[2,8,32],
                    "model.roi_head.bbox_head.0.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "model.roi_head.bbox_head.1.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "model.roi_head.bbox_head.2.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "classes":('char','word'), "data.train.classes":('char','word'), 
                    "data.val.classes":('char','word'), "data.test.classes":('char','word'),
                    "model.roi_head.bbox_head.0.num_classes": 2,
                    "model.roi_head.bbox_head.1.num_classes": 2,
                    "model.roi_head.bbox_head.2.num_classes": 2,
                    "model.roi_head.mask_head.num_classes": 2,
                }
            elif lang == "jp":
                loc_config = {
                    "model.rpn_head.anchor_generator.scales":[2,8,32],
                    "model.roi_head.bbox_head.0.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "model.roi_head.bbox_head.1.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "model.roi_head.bbox_head.2.norm_cfg.type": "BN" if device=="cpu" else "SyncBN",
                    "classes":('char',), "data.train.classes":('char',), 
                    "data.val.classes":('char',), "data.test.classes":('char',),
                    "model.roi_head.bbox_head.0.num_classes": 1,
                    "model.roi_head.bbox_head.1.num_classes": 1,
                    "model.roi_head.bbox_head.2.num_classes": 1,
                    "model.roi_head.mask_head.num_classes": 1,
                }
            else:
                raise NotImplementedError
            localizer = init_detector(localizer_config, localizer_checkpoint, device=device, cfg_options=loc_config)
        else:
            cfg = LazyConfig.load(localizer_config)
            if lang == "en":
                # cfg.model.roi_heads.num_classes=2
                # cfg.model.roi_heads.mask_head.num_classes=2
                cfg.train.init_checkpoint=localizer_checkpoint
            elif lang == "jp":
                # TODO address that these are two headed models as is...
                # cfg.model.roi_heads.num_classes=1
                # cfg.model.roi_heads.mask_head.num_classes=1
                cfg.train.init_checkpoint=localizer_checkpoint
            else:
                raise NotImplementedError

            # pp = pprint.PrettyPrinter(indent=2)
            # config_as_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            # pp.pprint(config_as_dict)

            localizer = instantiate(cfg.model)
            localizer.to(device)
            localizer = create_ddp_model(localizer)
            DetectionCheckpointer(localizer).load(cfg.train.init_checkpoint)
            localizer.eval()
        
        # load recognizer encoder

        recognizer_encoder = encoder.load(recognizer_checkpoint)
        recognizer_encoder.to(device)
        recognizer_encoder.eval()

        # configure recognizer

        if N_classes is None:
            knn_func = FaissKNN(
                index_init_fn=faiss.IndexFlatIP, 
                reset_before=False, reset_after=False
            )
            recognizer = InferenceModel(recognizer_encoder, knn_func=knn_func)

            if not ad_hoc_index_root_dir is None:
                render_dataset = create_render_dataset(
                    ad_hoc_index_root_dir,
                    lang=lang,
                    font_name="NotoSerifCJKjp-Regular" if lang == "jp" else "NotoSerif-Regular",
                    imsize=224
                )
                candidate_chars = [chr(int(os.path.basename(x[0]).split("_")[0], base=16)) if \
                    os.path.basename(x[0]).startswith("0x") else os.path.basename(x[0])[0] for x in render_dataset.data]
                candidate_chars_dict = {c:idx for idx, c in enumerate(candidate_chars)}
                print(f"{len(candidate_chars)} candidate chars!")
                recognizer.train_knn(render_dataset)
            else:
                with open(recognizer_chars) as f:
                    candidate_chars = f.read().split()
                    candidate_chars_dict = {c:idx for idx, c in enumerate(candidate_chars)}
                    print(f"{len(candidate_chars)} candidate chars!")
                recognizer.load_knn_func(recognizer_index)

            if not blacklist is None:
                blacklist_ids = np.array([candidate_chars_dict[blc] for blc in blacklist])
                recognizer.knn_func.index.remove_ids(blacklist_ids)
                candidate_chars = [c for c in candidate_chars if not c in blacklist]
            class_map_dict = None
        else:
            with open(class_map) as f:
                class_map_dict = json.load(f)
            recognizer = recognizer_encoder
            candidate_chars = None

        # set default args

        self.localizer = localizer
        self.recognizer = recognizer
        self.recongizer_encoder = recognizer_encoder
        self.vertical = vertical
        self.double_clipped = True
        self.candidate_chars = candidate_chars
        self.char_transform = char_transform
        self.save_chars = save_chars
        self.image_dir = image_dir
        self.score_thresh = score_thresh
        self.score_thresh_word = score_thresh_word
        self.spell_check = spell_check
        self.N_classes = N_classes
        self.class_map_dict = class_map_dict
        self.anchor_margin = anchor_margin
        self.lang = lang
        self.device = device
        self.LARGE_NUM = 1_000_000
        self.anchor_multiplier = 4
        self.knn = knn
        self.d2 = d2


    @staticmethod
    def mmdet_output_format(result):
        outputs = result[0]
        classes = outputs["instances"].pred_classes.tolist()
        boxes = outputs["instances"].pred_boxes.tensor.tolist()
        scores = outputs["instances"].scores.tolist()
        char_bboxes = [x + [scores[idx]] for idx, x in enumerate(boxes) if classes[idx]==0]
        word_bboxes = [x + [scores[idx]] for idx, x in enumerate(boxes) if classes[idx]==1]
        result = [[char_bboxes, word_bboxes]] if len(word_bboxes) > 0 else [[char_bboxes]]
        return result


    def infer(self, im):

        # localizer inference

        if not self.d2:
            result = inference_detector(self.localizer, im)
        else:
            pil_image = Image.open(im).convert("RGB")
            d2_image = np.moveaxis(np.array(pil_image), -1, 0)
            with torch.inference_mode():
                result = self.localizer([{'image': torch.from_numpy(d2_image)}])
            result = self.mmdet_output_format(result)

        # organize results of localizer inference

        if self.lang == "en":
            char_bboxes, word_bboxes = result if isinstance(result[0], np.ndarray) else result[0]
            char_bboxes, word_end_idx = self.en_preprocess(result)
        elif self.lang == "jp":
            char_bboxes, word_bboxes = self.jp_preprocess(result), None

        # get char crops for coordinates, store metadata about coordinates

        im = np.array(Image.open(im).convert("RGB"))
        im_height, im_width = im.shape[0], im.shape[1]
        char_crops = []
        charheights, charbottoms = [], []
        for bbox in char_bboxes:
            try:
                x0, y0, x1, y1 = map(int, map(round, bbox))
                if self.double_clipped: 
                    if self.vertical:
                        x0, y0, x1, y1 = 0, y0, im_width, y1
                    else:
                        x0, y0, x1, y1 = x0, 0, x1, im_height
                try:
                    char_crops.append(self.char_transform(im[y0:y1,x0:x1,:]))
                except ValueError:
                    print(x0, y0, x1, y1)
                    print("Value error")
                    exit(1)
                if self.lang == "en":
                    charheights.append(bbox[3]-bbox[1])
                    charbottoms.append(bbox[3])
            except (RuntimeError, IndexError):
                continue

        if len(char_crops) == 0:
            print("No content detected!")
            return None, None, None, None
        
        # perform batched recognizer inference

        if self.N_classes is None: # kNN

            with torch.no_grad():                  
                concat_char_dets = torch.stack(char_crops).to(self.device)
                char_det_square_emb = self.recongizer_encoder(concat_char_dets)

            char_det_all_concat = torch.nn.functional.normalize(char_det_square_emb, p=2, dim=1)
            _, indices = self.recognizer.knn_func(char_det_all_concat, k=self.knn)
            index_list = indices.squeeze(-1).cpu().tolist()
            nearest_chars = [[self.candidate_chars[nn] for nn in nns] for nns in index_list]

            if self.lang == "en":
                assert len(nearest_chars) == len(charheights) == len(charbottoms), \
                    f"{len(nearest_chars)} == {len(charheights)} == {len(charbottoms)}; {nearest_chars}"
                
        else: # FFNN
            
            with torch.no_grad():
                concat_char_dets = torch.stack(char_crops, dim=0).to(self.device)
                outputs = self.recognizer(concat_char_dets)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = logits.argmax(-1)
                predlist = predictions.detach().cpu().tolist()
            nearest_chars = [[self.class_map_dict[str(x)]] for x in predlist]
            
        # postprocessing (mostly for English)

        output_nns = ["".join(chars).strip() for chars in nearest_chars]
        output = "".join(x[0] for x in nearest_chars).strip()
        
        if self.lang == "en":
            output = self.en_postprocess(output, word_end_idx, charheights, charbottoms)
        
        return output, output_nns, char_bboxes, word_bboxes


    def en_preprocess(self, result):

        bboxes_char, bboxes_word = result if isinstance(result[0], np.ndarray) else result[0]
        sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_char = [x[:4] for x in sorted_bboxes_char if x[4] > self.score_thresh]
        sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_word = [x[:4] for x in sorted_bboxes_word if x[4] > self.score_thresh_word]

        word_end_idx = []
        closest_idx = 0
        sorted_bboxes_char_rights = [x[2] for x in sorted_bboxes_char]
        sorted_bboxes_word_lefts = [x[0] for x in sorted_bboxes_word]
        for wordleft in sorted_bboxes_word_lefts:
            prev_dist = self.LARGE_NUM
            for idx, charright in enumerate(sorted_bboxes_char_rights):
                dist = abs(wordleft-charright)
                if dist < prev_dist and charright > wordleft:
                    prev_dist = dist
                    closest_idx = idx
            word_end_idx.append(closest_idx)
        assert len(word_end_idx) == len(sorted_bboxes_word)

        return sorted_bboxes_char, word_end_idx


    def en_postprocess(self, line_output, word_end_idx, charheights, charbottoms):

        assert len(line_output) == len(charheights) == len(charbottoms), f"{len(line_output)} == {len(charheights)} == {len(charbottoms)}; {line_output}; {charbottoms}; {charheights}"

        if any(map(lambda x: len(x)==0, (line_output, word_end_idx, charheights, charbottoms))):
            return None

        outchars_w_spaces = [" " + x if idx in word_end_idx else x for idx, x in enumerate(line_output)]
        charheights_w_spaces = list(flatten([(self.LARGE_NUM, x) if idx in word_end_idx else x for idx, x in enumerate(charheights)]))
        charbottoms_w_spaces = list(flatten([(0, x) if idx in word_end_idx else x for idx, x in enumerate(charbottoms)]))
        charbottoms_w_spaces = charbottoms_w_spaces[1:] if charbottoms_w_spaces[0]==0 else charbottoms_w_spaces
        charheights_w_spaces = charheights_w_spaces[1:] if charheights_w_spaces[0]==self.LARGE_NUM else charheights_w_spaces

        line_output = "".join(outchars_w_spaces).strip()

        assert len(charheights_w_spaces) == len(line_output), \
            f"charheights_w_spaces = {len(charheights_w_spaces)}; output = {len(line_output)}; {charheights_w_spaces}; {line_output}"

        output_distinct_lower_idx = [idx for idx, c in enumerate(line_output) if c in create_distinct_lowercase()]

        if len(output_distinct_lower_idx) > 0 and not self.anchor_margin is None:
            avg_distinct_lower_height = sum(charheights_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
            output_tolower_idx = [idx for idx, c in enumerate(line_output) \
                if abs(charheights_w_spaces[idx] - avg_distinct_lower_height) < self.anchor_margin * avg_distinct_lower_height]
            output_toupper_idx = [idx for idx, c in enumerate(line_output) \
                if charheights_w_spaces[idx] - avg_distinct_lower_height > self.anchor_margin * self.anchor_multiplier * avg_distinct_lower_height]
            avg_distinct_lower_bottom = sum(charbottoms_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
            output_toperiod_idx = [idx for idx, c in enumerate(line_output) \
                if c == "-" and abs(charbottoms_w_spaces[idx] - avg_distinct_lower_bottom) < self.anchor_margin * avg_distinct_lower_height]

        if self.spell_check:
            line_output = visual_spell_checker(line_output, WORDDICT, SIMDICT, ABBREVSET)

        if len(output_distinct_lower_idx) > 0 and not self.anchor_margin is None:
            nondistinct_lower = create_nondistinct_lowercase()
            line_output = "".join([c.lower() if idx in output_tolower_idx else c for idx, c in enumerate(line_output)])
            line_output = "".join([c.upper() if idx in output_toupper_idx and c in nondistinct_lower else c for idx, c in enumerate(line_output)])
            line_output = "".join(["." if idx in output_toperiod_idx else c for idx, c in enumerate(line_output)])

        return line_output


    def jp_preprocess(self, result):

        bboxes_char = result[0][0]
        sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if self.vertical else x[0])
        sorted_bboxes_char = [x[:4] for x in sorted_bboxes_char if x[4] > self.score_thresh]

        return sorted_bboxes_char


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
        help="Path to directory of relevant image files")
    parser.add_argument("--coco_json", type=str,
        help="Path to COCO JSON specifying content of interest for OCRing")
    parser.add_argument("--recognizer_dir", type=str, required=True,
        help="Path to directory of recognizer materials, e.g., weights, char list, usually same as W&B run name")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "jp"],
        help="Language of interest")
    parser.add_argument("--vertical", action="store_true", default=False,
        help="Specify that the text input is vertical")
    parser.add_argument("--blacklist_chars", type=str, default=None,
        help="Blacklist chars, i.e., OCR will not recognize these chars in inference mode")
    parser.add_argument("--no_spaces_eval", action="store_true", default=False,
        help="Evaluate OCR results without regard for spaces")
    parser.add_argument("--device", type=str, default="cuda",
        help="Set device for model and data")
    parser.add_argument("--spell_check", action='store_true', default=False,
        help="Rule-based spell-checking for English")
    parser.add_argument("--norm_edit", action='store_true', default=False,
        help="Evaluate in terms of normalized edit distance, as opposed to CER")
    parser.add_argument("--localizer_dir", type=str, default=None,
        help="Path to directory with localizer materials, e.g., weights, configs")
    parser.add_argument("--rcnn_score_thr", type=float, default=0.3,
        help="Set RCNN head score threshold for detection for character objects")
    parser.add_argument("--rcnn_score_thr_word", type=float, default=0.3,
        help="Set RCNN head score threshold for detection for word objects, if applicable")
    parser.add_argument("--anchor_margin", type=float, default=None,
        help="Hyperparameter for English EffOCR post-processing")
    parser.add_argument("--infer_over_img_dir", action='store_true', default=False,
        help="Pass inputs a directory of images, no JSON, COCO or otherwise, required")
    parser.add_argument("--save_output", type=str, default=None,
        help="Save output to this directory")
    parser.add_argument('--N_classes', type=int, default=None,
        help="Triggers use of FFNN classifier head with N classes")
    parser.add_argument("--uncased", action='store_true', default=False,
        help="Evaluate OCR results uncased")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument("--ad_hoc_index_root_dir", type=str, default=None,
        help="Create render dataset used as an FAISS index ad hoc from this root dir")
    args = parser.parse_args()

    # create homoglyph dict and word set

    WORDDICT = create_worddict()
    SIMDICT = create_homoglyph_dict()
    ABBREVSET = create_common_abbrev()

    # open json

    if args.infer_over_img_dir:
        coco_images = glob(os.path.join(args.image_dir, "**/*.png"), recursive=True)
        coco_images += glob(os.path.join(args.image_dir, "**/*.jpg"), recursive=True)
    else:
        with open(args.coco_json) as f:
            coco = json.load(f)
        coco_images = [os.path.join(args.image_dir, x["file_name"]) for x in coco["images"]]

    # load encoder

    if args.auto_model_hf is None and args.auto_model_timm is None:
        raise NotImplementedError
    elif not args.auto_model_timm is None and args.N_classes is None:
        encoder = AutoEncoderFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None and args.N_classes is None:
        encoder = AutoEncoderFactory("hf", args.auto_model_hf)
    elif not args.auto_model_timm is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("timm", args.auto_model_timm, n_classes=args.N_classes)
    elif not args.auto_model_hf is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("hf", args.auto_model_hf, n_classes=args.N_classes)

    # create dataloader

    dataloader = create_dataset(coco_images, BASE_TRANSFORM)

    # create ocr engine

    loc_chkpt = os.path.join(args.localizer_dir, "best_bbox_mAP.pth") if \
        os.path.exists(os.path.join(args.localizer_dir, "best_bbox_mAP.pth")) \
            else os.path.join(args.localizer_dir, "model_best.pth")
    loc_cfg = glob(os.path.join(args.localizer_dir, "*.py"))[0] if \
        len(glob(os.path.join(args.localizer_dir, "*.py"))) > 0 else \
            os.path.join(args.localizer_dir, "config.yaml")

    d2 = loc_cfg == os.path.join(args.localizer_dir, "config.yaml")

    ocr_engine = EffOCR(
        localizer_checkpoint=loc_chkpt,
        localizer_config=loc_cfg,
        recognizer_checkpoint=os.path.join(args.recognizer_dir, "enc_best.pth"), 
        recognizer_index=os.path.join(args.recognizer_dir, "ref.index"),
        recognizer_chars=os.path.join(args.recognizer_dir, "ref.txt"),
        class_map=os.path.join(args.recognizer_dir, "class_map.json"),
        encoder=encoder,
        image_dir=args.image_dir,
        vertical=args.vertical,
        lang=args.lang,
        device=args.device,
        char_transform=create_paired_transform(),
        anchor_margin=args.anchor_margin,
        blacklist=args.blacklist_chars,
        score_thresh=args.rcnn_score_thr,
        score_thresh_word=args.rcnn_score_thr_word,
        spell_check=args.spell_check,
        N_classes=args.N_classes,
        d2=loc_cfg == os.path.join(args.localizer_dir, "config.yaml"),
        ad_hoc_index_root_dir=args.ad_hoc_index_root_dir
    )

    # param count

    localizer_params = count_parameters(ocr_engine.localizer)
    recognizer_params = count_parameters(ocr_engine.recongizer_encoder)
    print(f"Total trainable parameters for EffOCR: {localizer_params + recognizer_params}")
    
    # perform inference

    inference_results = {}
    inference_coco = copy.deepcopy(COCO_JSON_SKELETON)
    image_id, anno_id = 0, 0

    with torch.no_grad():
        for path in tqdm(coco_images):
            # input = input.cuda()
            # path, = path
            W, H = Image.open(path).size
            output, nn_output, char_boxes, word_boxes = ocr_engine.infer(path)
            
            if output is None:
                continue
            assert len(nn_output) == len(char_boxes) == len(output.replace(" ", "")), f"{char_boxes}"
            if args.lang == "jp":
                inference_coco["images"].append(create_coco_image_entry(os.path.basename(path), H, W, image_id, text=output))
                for nnchars, charbox in zip(nn_output, char_boxes):
                    x0, y0, x1, y1 = map(int, map(round, charbox))
                    x, y, w, h = x0, y0, x1 - x0, y1 - y0
                    inference_coco["annotations"].append(create_coco_anno_entry(x, y, w, h, anno_id, image_id, cat_id=0, text=nnchars))
            inference_results[path] = output
            image_id += 1

    # optionally save output and end script

    if args.save_output:
        os.makedirs(args.save_output, exist_ok=True)
        os.makedirs(os.path.join(args.save_output, "images"), exist_ok=True)
        for im in coco_images:
            Image.open(im).save(os.path.join(args.save_output, "images", os.path.basename(im)))
        with open(os.path.join(args.save_output, "inference_results.json"), "w") as f:
            json.dump(inference_results, f, indent=2)
        with open(os.path.join(args.save_output, "inference_coco.json"), "w") as f:
            json.dump(inference_coco, f, indent=2)
        exit(0)
    
    inference_results = {os.path.basename(k): v for k, v in inference_results.items()}

    # collect ground truth transcriptions and associate ground truth with predictions

    gts = []
    for x in coco["images"]:
        filename = x["file_name"]
        gt_chars = x["text"]
        gts.append((filename, gt_chars))

    gt_pred_pairs = gt_collect(inference_results, gts)

    # print results

    acc, norm_ED = textline_evaluation(gt_pred_pairs, print_incorrect=False, 
        no_spaces_in_eval=args.no_spaces_eval, norm_edit_distance=args.norm_edit, uncased=args.uncased)
    print(f"EffOCR | Textline accuracy = {acc} | CER = {norm_ED}")

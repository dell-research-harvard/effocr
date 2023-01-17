import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_metric_learning.utils.inference import FaissKNN
from torchvision import transforms as T
import faiss
from tqdm import tqdm
import json
import argparse
import numpy as np
import queue
import threading
from glob import glob
import os
import sys
import io
import requests
import base64
import copy
from PIL import Image
# import omegaconf
import time

# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import LazyConfig, instantiate
# from detectron2.engine.defaults import create_ddp_model
# from detectron2.data import MetadataCatalog
# from mmdet.apis import init_detector, inference_detector
# from mmdet.datasets import replace_ImageToTensor
import mmcv
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/issues/59

from deepsparse import compile_model
from deepsparse.pipelines.custom_pipeline import CustomTaskPipeline

sys.path.insert(0, "../")
from utils.datasets_utils import *
from effocr_datasets.inference_datasets import *
from utils.eval_utils import *
from utils.coco_utils import *
from utils.spell_check_utils import *
from onnx_engines.localizer_engine import EffLocalizer
from onnx_engines.recognizer_engine import EffRecognizer


LARGE_NUMBER = 1_000_000_000


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


def en_preprocess(bboxes_char, bboxes_word, vertical=False):

    sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if vertical else x[0])
    sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if vertical else x[0])

    word_end_idx = []
    closest_idx = 0
    sorted_bboxes_char_rights = [x[2] for x in sorted_bboxes_char]
    sorted_bboxes_word_lefts = [x[0] for x in sorted_bboxes_word]
    for wordleft in sorted_bboxes_word_lefts:
        prev_dist = LARGE_NUMBER
        for idx, charright in enumerate(sorted_bboxes_char_rights):
            dist = abs(wordleft-charright)
            if dist < prev_dist and charright > wordleft:
                prev_dist = dist
                closest_idx = idx
        word_end_idx.append(closest_idx)
    assert len(word_end_idx) == len(sorted_bboxes_word)

    return sorted_bboxes_char, word_end_idx


def en_postprocess(line_output, word_end_idx, charheights, charbottoms, anchor_margin=None, anchor_multiplier = 4):

    assert len(line_output) == len(charheights) == len(charbottoms), f"{len(line_output)} == {len(charheights)} == {len(charbottoms)}; {line_output}; {charbottoms}; {charheights}"

    if any(map(lambda x: len(x)==0, (line_output, word_end_idx, charheights, charbottoms))):
        return None

    outchars_w_spaces = [" " + x if idx in word_end_idx else x for idx, x in enumerate(line_output)]
    charheights_w_spaces = list(flatten([(LARGE_NUMBER, x) if idx in word_end_idx else x for idx, x in enumerate(charheights)]))
    charbottoms_w_spaces = list(flatten([(0, x) if idx in word_end_idx else x for idx, x in enumerate(charbottoms)]))
    charbottoms_w_spaces = charbottoms_w_spaces[1:] if charbottoms_w_spaces[0]==0 else charbottoms_w_spaces
    charheights_w_spaces = charheights_w_spaces[1:] if charheights_w_spaces[0]==LARGE_NUMBER else charheights_w_spaces

    line_output = "".join(outchars_w_spaces).strip()

    assert len(charheights_w_spaces) == len(line_output), \
        f"charheights_w_spaces = {len(charheights_w_spaces)}; output = {len(line_output)}; {charheights_w_spaces}; {line_output}"

    output_distinct_lower_idx = [idx for idx, c in enumerate(line_output) if c in create_distinct_lowercase()]

    if len(output_distinct_lower_idx) > 0 and not anchor_margin is None:
        avg_distinct_lower_height = sum(charheights_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
        output_tolower_idx = [idx for idx, c in enumerate(line_output) \
            if abs(charheights_w_spaces[idx] - avg_distinct_lower_height) < anchor_margin * avg_distinct_lower_height]
        output_toupper_idx = [idx for idx, c in enumerate(line_output) \
            if charheights_w_spaces[idx] - avg_distinct_lower_height > anchor_margin * anchor_multiplier * avg_distinct_lower_height]
        avg_distinct_lower_bottom = sum(charbottoms_w_spaces[idx] for idx in output_distinct_lower_idx) / len(output_distinct_lower_idx)
        output_toperiod_idx = [idx for idx, c in enumerate(line_output) \
            if c == "-" and abs(charbottoms_w_spaces[idx] - avg_distinct_lower_bottom) < anchor_margin * avg_distinct_lower_height]

    # if self.spell_check:
    #     line_output = visual_spell_checker(line_output, WORDDICT, SIMDICT, ABBREVSET)

    if len(output_distinct_lower_idx) > 0 and not anchor_margin is None:
        nondistinct_lower = create_nondistinct_lowercase()
        line_output = "".join([c.lower() if idx in output_tolower_idx else c for idx, c in enumerate(line_output)])
        line_output = "".join([c.upper() if idx in output_toupper_idx and c in nondistinct_lower else c for idx, c in enumerate(line_output)])
        line_output = "".join(["." if idx in output_toperiod_idx else c for idx, c in enumerate(line_output)])

    return line_output


def jp_preprocess(bboxes_char, vertical=True):
    try:
        sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if vertical else x[0])
    except TypeError:
        print(bboxes_char)
        exit(1)
    return sorted_bboxes_char


def create_batches(data, batch_size = 64):
    """Create batches for inference"""

    batches = []
    batch = []
    for i, d in enumerate(data):
        if d is not None:
            batch.append(d)
        else:
            batch.append(torch.zeros((3, 224, 224)))
        if (i+1) % batch_size == 0:
            batches.append(torch.stack(batch))
            batch = []
    if len(batch) > 0:
        batches.append(torch.nn.functional.pad(torch.stack(batch), (0, 0, 0, 0, 0, 0, 0, 64 - len(batch))))
    return [b.detach().numpy() for b in batches]


def iteration(model, input):
    output = model.run(input)
    return output, output


class LocalizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(LocalizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            path = self._input_queue.get()
            output = iteration(self._model, [path])
            self._output_queue.put((path, output))


class TransformationThread(threading.Thread):
    def __init__(
        self,
        transformation,
        input_queue: queue.Queue,
        output_queue: queue.Queue
    ):
        super(TransformationThread, self).__init__()
        self.transformation = transformation
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while not self.input_queue.empty():
            i, char = self.input_queue.get()
            try:
                output = self.transformation(char)
                self.output_queue.put((i, output))
            except:
                self.output_queue.put((i, None))


class RecognizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(RecognizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            i, batch = self._input_queue.get()
            output = iteration(self._model, batch)
            self._output_queue.put((i, output))


# @profile
def run_effocr(coco_images, localizer_engine, recognizer_engine, char_transform, lang, num_streams=4, 
                            vertical=False, localizer_output = None, conf_thres=0.5):
    start_time = time.time()
    inference_results = {}
    inference_coco = copy.deepcopy(COCO_JSON_SKELETON)
    image_id, anno_id = 0, 0
    
    input_queue = queue.Queue()
    for p in coco_images:
        input_queue.put(p)
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(LocalizerEngineExecutorThread(localizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    char_crops, word_end_idxs, n_chars = [], [], []
    charheights, charbottoms, coco_new_order = [], [], []

    while not output_queue.empty():
        path, result = output_queue.get()
        coco_new_order.append(path)
        
        if localizer_engine._model_backend == 'yolo':
            result = result[0][0]
            bboxes, labels = result[:, :4], result[:, -1]
        elif localizer_engine._model_backend == 'detectron2':
            result = result[0][0]
            bboxes, labels = result[0][result[3] > conf_thres], result[1][result[3] > conf_thres]
            bboxes, labels = torch.from_numpy(bboxes), torch.from_numpy(labels)
        elif localizer_engine._model_backend == 'mmdetection':
            result = result[0][0]
            bboxes, labels = result[0][result[0][:, -1] > conf_thres], result[1][result[0][:, -1] > conf_thres]
            bboxes = bboxes[:, :-1]
            bboxes, labels = torch.from_numpy(bboxes), torch.from_numpy(labels)

        print(bboxes.size())
        print(labels.size())
        if lang == "en":
            char_bboxes, word_bboxes = bboxes[labels == 0], bboxes[labels == 1]

            if len(char_bboxes) != 0:
                char_bboxes, word_end_idx = en_preprocess(char_bboxes, word_bboxes)
                n_chars.append(len(char_bboxes))
                word_end_idxs.append(word_end_idx)
            else:
                n_chars.append(0)
                word_end_idxs.append([])
        elif lang == "jp":
            char_bboxes = bboxes[labels == 0] #there should be no other boxes, but have this just in case
            if len(char_bboxes) != 0:
                char_bboxes = jp_preprocess(char_bboxes, vertical=vertical)
                n_chars.append(len(char_bboxes))
            else:
                n_chars.append(0)

        print(n_chars)
        print(word_end_idxs)
        
        if localizer_output:
            img = Image.open(path).convert("RGB")
            im_width, im_height = img.size[0], img.size[1]
            # print(im_height, im_width)
            draw = ImageDraw.Draw(img)
            for bbox in char_bboxes:
                x0, y0, x1, y1 = torch.round(bbox)
                if vertical:
                    x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
                else:
                    x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                draw.rectangle((x0, y0, x1, y1), outline="red")
            img.save(os.path.join(localizer_output, os.path.basename(path)))

        im = np.array(Image.open(path).convert("RGB"))
        im_height, im_width = im.shape[0], im.shape[1]

        for bbox in char_bboxes:
            x0, y0, x1, y1 = torch.round(bbox)

            if vertical:
                x0, y0, x1, y1 = 0, int(round(y0.item() * im_height / 640)), im_width, int(round(y1.item() * im_height / 640))
            else:
                x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

            char_crops.append(im[y0:y1, x0:x1, :])

            if lang == "en":
                charheights.append(bbox[3]-bbox[1])
                charbottoms.append(bbox[3])

    print(len(char_crops))

    #Now transform the char crops as a batch
    input_queue = queue.Queue()
    for i, batch in enumerate(char_crops):
        input_queue.put((i, batch))
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(TransformationThread(char_transform, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    char_crops = [None] * len(char_crops)
    while not output_queue.empty():
        i, char = output_queue.get()
        char_crops[i] = char

    char_crop_batches = create_batches(char_crops)
    print(len(char_crop_batches))

    #Now run the transformed crops through the recognizer, including knn
    input_queue = queue.Queue()
    for i, batch in enumerate(char_crop_batches):
        input_queue.put((i, batch))
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    embeddings = [None] * len(char_crop_batches)
    while not output_queue.empty():
        i, result = output_queue.get()
        embeddings[i] = result

    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding[0][0]), p=2, dim=1) for embedding in embeddings]
    indices = [knn_func(embedding, k=1)[1] for embedding in embeddings]
    index_list = [index.squeeze(-1).tolist() for index in indices]
    indices = [item for sublist in index_list for item in sublist]
    nn_outputs = [candidate_chars[idx] for idx in indices]

    #Now run postprocessing
    idx, textline_outputs, textline_bottoms, textline_heights = 0, [], [], []
    for l in n_chars:
        textline_outputs.append(nn_outputs[idx:idx+l])
        textline_bottoms.append(charbottoms[idx:idx+l])
        textline_heights.append(charheights[idx:idx+l])
        idx += l

    output_nns = [["".join(chars).strip() for chars in textline] for textline in textline_outputs]
    outputs = ["".join(x[0] for x in textline).strip() for textline in textline_outputs]

    if lang == "en":
        for i, path in enumerate(coco_new_order):
            inference_results[path] = en_postprocess(outputs[i], word_end_idxs[i], textline_heights[i], textline_bottoms[i])
    else:
        for i, path in enumerate(coco_new_order):
            inference_results[path] = outputs[i]
    print('Total time: {:.2f}s'.format(time.time() - start_time))
    print('Average time per image: {:.4f}s'.format((time.time() - start_time) / len(coco_images)))

    return inference_results, inference_coco


if __name__ == "__main__":
    '''
    Notes on specific Arguments

    - image_dir: must contain all image files referenced in the coco_json file
    - coco_json: must be in coco format
    - recognizer_dir: must contain all of the following:
            'enc_best.pth' OR 'enc_best.onnx' -- The actual recognizer model or its ONNX version
            'ref.index' -- Reference embeddings for KNN
            'ref.txt' -- Mapping from embeddings in ref.index to actual characters
            'class_map.json' -- ONLY needed if explicitly providing a number of classes in the N_classes argument,
                                overrides a class mapping created directly from the ref files above
    - lang: select en or jp (only en support for ONNX/NM currently)
    - vertical: what direction does the text go?
    - no_effocr: Only run selected tesseract/gcv/baidu comparison, do not actually run effocr
    - localizer_dir: must contain all of the following:
            'best_bbox_mAP.pth' OR 'best_bbox_mAP.onnx' OR (if using a detectron2 model) 'model_best.pth' OR 'model_best.onnx'
                                    -- The actual localizer model or its ONNX version
            '*.py' or 'config.yaml' the config file for the localizer model (must be only python file in the directory)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
        help="Path to directory of relevant image files")
    parser.add_argument("--coco_json", type=str,
        help="Path to COCO JSON specifying textlines of interest for OCRing")
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
    parser.add_argument("--spell_check", action='store_true', default=False,
        help="Rule-based spell-checking for English")
    parser.add_argument("--norm_edit", action='store_true', default=False,
        help="Evaluate in terms of normalized edit distance, as opposed to CER")
    parser.add_argument("--localizer_dir", type=str, default=None,
        help="Path to directory with localizer materials, e.g., weights, configs")
    parser.add_argument('--localizer_iou_thresh', type=float, default=0.01,
        help="IOU threshold for localizer")
    parser.add_argument('--localizer_conf_thresh', type=float, default=0.35,
        help="Confidence threshold for localizer")
    parser.add_argument("--anchor_margin", type=float, default=None,
        help="Hyperparameter for English EffOCR post-processing")
    parser.add_argument("--infer_over_img_dir", action='store_true', default=False,
        help="Pass inputs a directory of images, no JSON, COCO or otherwise, required")
    parser.add_argument("--save_output", type=str, default=None,
        help="Save output!")
    parser.add_argument("--uncased", action='store_true', default=False,
        help="Evaluate OCR results uncased")
    parser.add_argument('--num_threads', type=int, default=None)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--output_localizer_results', default=None)
    parser.add_argument('--backend', type=str, default='yolo', choices=['yolo', 'detectron2', 'mmdetection'])
    parser.add_argument('--localizer_input_shape', type=int, nargs=2, default=None)
    args = parser.parse_args()

    # create homoglyph dict and word set
    if args.spell_check:
        WORDDICT = create_worddict()
        SIMDICT = create_homoglyph_dict()
        ABBREVSET = create_common_abbrev()

    if args.infer_over_img_dir:
        coco_images = glob(os.path.join(args.image_dir, "**/*.png"), recursive=True)
        coco_images += glob(os.path.join(args.image_dir, "**/*.jpg"), recursive=True)
    else:
        with open(args.coco_json) as f:
            coco = json.load(f)
        coco_images = [os.path.join(args.image_dir, x["file_name"]) for x in coco["images"]]

    # confirm that the ONNX models we're expecting are in fact present:
    assert os.path.exists(os.path.join(args.recognizer_dir, 'enc_best.onnx')), 'Recognizer model not found! Should be in {}/enc_best.onnx'.format(args.recognizer_dir)
    assert os.path.exists(os.path.join(args.localizer_dir, 'best_bbox_mAP.onnx')), 'Localizer model not found! Should be in {}/best_bbox_mAP.onnx'.format(args.localizer_dir)

    #Create localizer engine
    localizer_engine = EffLocalizer(
        os.path.join(args.localizer_dir, 'best_bbox_mAP.onnx'),
        iou_thresh=args.localizer_iou_thresh,
        conf_thresh=args.localizer_conf_thresh,
        vertical=args.vertical,
        num_cores=args.num_threads,
        model_backend=args.backend,
        input_shape=args.localizer_input_shape
    )

    char_transform = create_paired_transform(lang=args.lang)

    recognizer_engine = EffRecognizer(
        model = os.path.join(args.recognizer_dir, 'enc_best.onnx'),
        num_cores=args.num_threads
    )

    knn_func = FaissKNN(
        index_init_fn=faiss.IndexFlatIP,
        reset_before=False, reset_after=False
    )
    knn_func.load(os.path.join(args.recognizer_dir, "ref.index"))

    with open(os.path.join(args.recognizer_dir, "ref.txt")) as f:
        candidate_chars = f.read().split()
        candidate_chars_dict = {c:idx for idx, c in enumerate(candidate_chars)}
        print(f"{len(candidate_chars)} candidate chars!")

    if not args.blacklist_chars is None:
        blacklist_ids = np.array([candidate_chars_dict[blc] for blc in args.blacklist_chars])
        knn_func.index.remove_ids(blacklist_ids)
        candidate_chars = [c for c in candidate_chars if not c in args.blacklist_chars]

    coco_images = [c for c in coco_images for _ in range(args.n_repeats)]

    # run the effocr pipeline
    inference_results, inference_coco = run_effocr(coco_images, localizer_engine, recognizer_engine, 
                                                char_transform, args.lang, num_streams = args.num_threads, vertical=args.vertical,
                                                localizer_output = args.output_localizer_results, conf_thres=args.localizer_conf_thresh)


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

    inference_results = {os.path.basename(k): v for k, v in inference_results.items()}

    # collect ground truth transcriptions and associate ground truth with predictions

    gts = []
    for x in coco["images"]:
        filename = x["file_name"]
        gt_chars = x["text"]
        gts.append((filename, gt_chars))

    gt_pred_pairs = gt_collect(inference_results, gts)

    # print results
    acc, norm_ED = textline_evaluation(gt_pred_pairs, print_incorrect=True,
            no_spaces_in_eval=args.no_spaces_eval, norm_edit_distance=args.norm_edit, uncased=args.uncased)
    print(f"EffOCR | Textline accuracy = {acc} | CER = {norm_ED}")

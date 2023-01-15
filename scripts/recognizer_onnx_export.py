# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:20:45 2022

@author: bryan
"""
import numpy as np
import argparse
import torch
import onnx
from PIL import Image
import onnxruntime as ort
import os
import json
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.encoders import *
from models.classifiers import *
from datasets.recognizer_datasets import *
from utils.datasets_utils import *


def create_onnx_export(query_paths, model, transform, batch_size=64, seed=111, query_txt=False, onnx_path=None, quiet =False):

    def get_img_from_query_path(query_path):
        query_stem, _ = os.path.splitext(os.path.basename(query_path))
        query_img = Image.open(query_path).convert("RGB")
        query = transform(query_img).unsqueeze(0)
        return query

    def create_query_batches(query_paths, batch_size):
        batch, batches = [], []
        for q in query_paths:
            if len(batch) == batch_size:
                batches.append(torch.concat(batch))
                batch = []
            batch.append(get_img_from_query_path(q))
        return batches

    # optionally perform inference on text list of paths to images

    if query_txt:
        with open(query_paths) as f:
            query_paths = f.read().split()

    # randomness

    np.random.seed(seed)
    np.random.shuffle(query_paths)

    #create query batches
    batches = create_query_batches(query_paths, batch_size)

    model.eval()

    export_sample = batches[0].to('cuda')
    test_sample = batches[1].to('cuda')

    export_sample_res = model(export_sample).detach().cpu().numpy()
    test_sample_res = model(test_sample).detach().cpu().numpy()

    torch.onnx.export(model,
                      export_sample,
                      onnx_path,
                      dynamic_axes= {'imgs': [0]},
                      input_names=['imgs'],
                      output_names=['embs'],
                      opset_version=11)

    if not quiet:
        print('Exported model to {}'.format(onnx_path))

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    onnx_ses = ort.InferenceSession(onnx_path)
    outputs = onnx_ses.run(None, {'imgs': export_sample.detach().cpu().numpy()})
    outputs_2 = onnx_ses.run(None, {'imgs': test_sample.detach().cpu().numpy()})

    np.testing.assert_allclose(export_sample_res, outputs[0], rtol= .05, atol= .05)
    if not quiet:
        print('Output Sample Matches!')
    np.testing.assert_allclose(test_sample_res, outputs_2[0], rtol= .05, atol= .05)
    if not quiet:
        print('Test Sample Matches!')

    # print('Export sample Torch Model:', export_sample_res[:2, :50])
    # print('Export sample Sparse Model:', outputs[0][:2, :50])
    # print('Test sample Torch ModelL', test_sample_res[:2, :50])
    # print('Test sample Sparse Model:', outputs_2[0][:2, :50])

def onnx_export(torch_model, ref_dir_path, anno_path, query_text, auto_model_hf, auto_model_timm,
                lang, batch_size, onnx_path, quiet):
    #Set up encoder
    if not auto_model_timm is None:
        encoder = AutoEncoderFactory("timm", auto_model_timm)
    elif not auto_model_hf is None:
        encoder = AutoEncoderFactory("hf", auto_model_hf)
    else:
        raise ValueError('Must provide either a timm or hf encoder to build off of!')

    #Create transform
    transform = create_paired_transform(lang)

    # load best checkpoint
    encoder = encoder(auto_model_timm)
    encoder = encoder.load(torch_model)


    if lang != 'en':
        raise NotImplementedError('Only English characters are currently supported!')

    # construct separate datasets for paired and rendered images

    paired_dataset = create_paired_dataset(ref_dir_path, lang)
    render_dataset = create_render_dataset(ref_dir_path, lang,
        font_name="NotoSerifCJKjp-Regular" if lang == "jp" else "NotoSerif-Regular")


    with open(anno_path) as f:
        coco_anno = json.load(f)
        seg_ids = [os.path.splitext(x['file_name'])[0] for x in coco_anno['images']]
        query_paths = [x[0] for x in paired_dataset.data if any(f"PAIRED_{y}_" in x[0] for y in seg_ids)]

    create_onnx_export(
        query_paths,
        encoder,
        transform,
        batch_size=batch_size,
        onnx_path=onnx_path,
        quiet = quiet
    )

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-model', type=str, required=True,
        help='Path to model to convert')
    parser.add_argument("--ref_dir_path", type=str, default="./ref_dir",
        help="Root image directory path, with character class subfolders")
    parser.add_argument("--anno_path", type=str,
        help="Path to train or test or val annotations")
    parser.add_argument('--query_text', action='store_true', default=False,
        help="Inference on images declared via paths in a text file")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument('--lang', type=str, default='en',
        help='Character language: supported are en and jp')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--onnx_path', default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    onnx_export(args.torch_model,
                args.ref_dir_path,
                args.anno_path,
                args.query_text,
                args.auto_model_hf,
                args.auto_model_timm,
                args.lang,
                args.batch_size,
                args.onnx_path,
                args.quiet)

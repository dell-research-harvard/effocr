import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel, 
    TrOCRProcessor,
    VisionEncoderDecoderConfig
)
from tqdm import tqdm
import json
import argparse
import os

from utils.eval_utils import *


class ImageTextPairDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=256):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True,
        help="Path to COCO JSON file for assessment")
    parser.add_argument("--image_dir", type=str, required=True,
        help="Path to relevant image directory")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size")
    parser.add_argument("--pt_model", type=str, default="microsoft/trocr-base-stage1",
        help="Pretrained TrOCR from HF hub")
    parser.add_argument("--local_dir", type=str, default=None,
        help="Pretrained TrOCR from local dir")
    args = parser.parse_args()

    # load coco jsons
    with open(args.coco_json) as f:
        coco = json.load(f)

    # create lists of tuples of (file name, text ground truth)
    coco_path_gt_pairs = [(x["file_name"], x["text"]) for x in coco["images"]]

    # convert to dataframes
    df = pd.DataFrame(coco_path_gt_pairs, columns=['file_name', 'text'])
    df.reset_index(drop=True, inplace=True)

    # create datasets
    processor = TrOCRProcessor.from_pretrained(args.pt_model)
    dataset = ImageTextPairDataset(root_dir=args.image_dir, df=df, processor=processor)
    print("Number of test examples:", len(dataset))

    # create loaders
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # set device and load model to it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_dir is None:
        model = VisionEncoderDecoderModel.from_pretrained(args.pt_model)
    else:
        encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained(args.local_dir)
        model = VisionEncoderDecoderModel.from_pretrained(args.local_dir, config=encoder_decoder_config)
    model.to(device)
    count_parameters(model)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # load model and evaluate on test set
    pairs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to(device))
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            label_ids = batch["labels"]
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            pairs.extend(list(zip(label_str, pred_str)))

    best_custom_accuracy, best_custom_cer = textline_evaluation(
        pairs, print_incorrect=True, 
        no_spaces_in_eval=False, 
        norm_edit_distance=False, 
        uncased=True
    )

    print(f"Final CER: {best_custom_cer}")

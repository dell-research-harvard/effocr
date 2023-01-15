import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim import AdamW
from transformers import (
    VisionEncoderDecoderModel, 
    TrOCRProcessor,
    VisionEncoderDecoderConfig
)

from tqdm import tqdm
import json
import wandb
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
    parser.add_argument("--train_coco_json", type=str, required=True,
        help="Path to COCO JSON file with training data")
    parser.add_argument("--test_coco_json", type=str, required=True,
        help="Path to COCO JSON file with test data")
    parser.add_argument("--val_coco_json", type=str, required=True,
        help="Path to COCO JSON file with validation data")
    parser.add_argument("--image_dir", type=str, required=True,
        help="Path to relevant image directory")
    parser.add_argument("--run_name", type=str, required=True,
        help="Name of run for W&B logging purposes")
    parser.add_argument("--num_epochs", type=int, default=10,
        help="Number of epochs to train model")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
        help="Learning rate")
    parser.add_argument("--zero_shot", action="store_true", default=False,
        help="Run zero-shot inference")
    parser.add_argument("--pt_model", type=str, default="microsoft/trocr-base-stage1",
        help="Pretrained TrOCR from HF hub")
    parser.add_argument("--local_dir", type=str, default=None,
        help="Pretrained TrOCR from local dir")
    args = parser.parse_args()

    # initialize project

    wandb.init(project="trocr_v3", name=args.run_name)
    os.makedirs(args.run_name, exist_ok=True)

    # load coco jsons

    with open(args.train_coco_json) as f:
        coco_train = json.load(f)
    with open(args.test_coco_json) as f:
        coco_test = json.load(f)
    with open(args.val_coco_json) as f:
        coco_val = json.load(f)

    # create lists of tuples of (file name, text ground truth)

    coco_train_path_gt_pairs = [(x["file_name"], x["text"]) for x in coco_train["images"]]
    coco_test_path_gt_pairs = [(x["file_name"], x["text"]) for x in coco_test["images"]]
    coco_val_path_gt_pairs = [(x["file_name"], x["text"]) for x in coco_val["images"]]

    # convert to dataframes

    train_df = pd.DataFrame(coco_train_path_gt_pairs, columns=['file_name', 'text'])
    test_df = pd.DataFrame(coco_test_path_gt_pairs, columns=['file_name', 'text'])
    val_df = pd.DataFrame(coco_val_path_gt_pairs, columns=['file_name', 'text'])
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # create datasets

    processor = TrOCRProcessor.from_pretrained(args.pt_model)
    train_dataset = ImageTextPairDataset(root_dir=args.image_dir, df=train_df, processor=processor)
    test_dataset = ImageTextPairDataset(root_dir=args.image_dir, df=test_df, processor=processor)
    val_dataset = ImageTextPairDataset(root_dir=args.image_dir, df=val_df, processor=processor)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    print("Number of test examples:", len(test_dataset))

    # create loaders

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) if not args.zero_shot else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # set device and load model to it

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_dir is None:
        model = VisionEncoderDecoderModel.from_pretrained(args.pt_model)
    else:
        encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained(args.local_dir)
        model = VisionEncoderDecoderModel.from_pretrained(args.local_dir, config=encoder_decoder_config)
       
    model.to(device)

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

    # set other hyperparameters
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    count_parameters(model)
    model = torch.nn.DataParallel(model)
    best_cer = 100

    # training loop!

    if not args.zero_shot:

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times

            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                # get the inputs
                for k,v in batch.items():
                    batch[k] = v.to(device)

                # forward + backward + optimize
                outputs = model(**batch)
                loss = outputs.loss
                loss.sum().backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.sum().item()
            
            print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
            wandb.log({"train/loss": train_loss/len(train_dataloader)})
            
            # evaluate
            model.eval()

            # valid_cer = 0.0
            pairs = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                    # run batch generation
                    outputs = model.module.generate(batch["pixel_values"].to(device))
                    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
                    label_ids = batch["labels"]
                    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
                    pairs.extend(list(zip(label_str, pred_str)))

            custom_accuracy, custom_cer = textline_evaluation(
                pairs, print_incorrect=False, 
                no_spaces_in_eval=False, 
                norm_edit_distance=False, 
                uncased=True
            )

            if custom_cer < best_cer:
                best_cer = custom_cer
                model.module.save_pretrained(args.run_name)

            print("Validation CER custom:", custom_cer)
            print("Validation accuracy:", custom_accuracy)
            print("***")
            wandb.log({"val/cer": custom_cer})

    # load best model and evaluate on test set

    best_model = VisionEncoderDecoderModel.from_pretrained(args.run_name if not args.zero_shot else args.pt_model)
    best_model.to(device)
    best_model.eval()
    pairs = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # run batch generation
            outputs = best_model.generate(batch["pixel_values"].to(device))
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

    wandb.log({"test/cer": best_custom_cer})
    print(f"Final CER: {best_custom_cer}")

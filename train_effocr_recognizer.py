import torch
import torch.nn as nn
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import logging
import faiss
import os
from torchvision import transforms as T
from torch.nn import CrossEntropyLoss
import numpy as np

logging.getLogger().setLevel(logging.INFO)
from transformers import AdamW
import wandb
import argparse
from collections import defaultdict

from models.encoders import *
from models.classifiers import *
from datasets.recognizer_datasets import * # make sure Huggingface datasets is not installed...
from utils.datasets_utils import INV_NORMALIZE


def infer_hardneg(query_paths, ref_dataset, model, index_path, transform, inf_save_path, k=8, finetune=False):

    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
    infm = InferenceModel(model, knn_func=knn_func)
    infm.load_knn_func(index_path)
    
    all_nns = []
    for query_path in query_paths:
        im = Image.open(query_path).convert("RGB")
        query = transform(im).unsqueeze(0)
        _, indices = infm.get_nearest_neighbors(query, k=k)
        nn_chars = []
        for i in indices[0]:
            path_elements = os.path.basename(ref_dataset.data[i][0]).split("_")
            nn_chars.append(path_elements[-2] if finetune else path_elements[0])
        nn_chars = [chr(int(c, base=16)) if c.startswith("0x") else c for c in nn_chars]
        all_nns.append("".join(nn_chars))

    with open(inf_save_path, 'w') as f:
        f.write("\n".join(all_nns))


def save_ref_index(ref_dataset, model, save_path):

    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
    infm = InferenceModel(model, knn_func=knn_func)
    infm.train_knn(ref_dataset)
    infm.save_knn_func(os.path.join(save_path, "ref.index"))

    ref_data_file_names = []
    for x in ref_dataset.data:
        if os.path.basename(x[0]).startswith("0x"):
            ref_data_file_names.append(chr(int(os.path.basename(x[0]).split("_")[0], base=16)))
        else:
            ref_data_file_names.append(os.path.basename(x[0])[0])

    with open(os.path.join(save_path, "ref.txt"), "w") as f:
        f.write("\n".join(ref_data_file_names))


def save_model(model_folder, enc, epoch, datapara):

    if not os.path.exists(model_folder): os.makedirs(model_folder)

    if datapara:
        torch.save(enc.module.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))
    else:
        torch.save(enc.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))


def get_all_embeddings(dataset, model, batch_size=128):

    tester = testers.BaseTester(batch_size=batch_size)
    return tester.get_all_embeddings(dataset, model)


def tester_knn(test_set, ref_set, model, accuracy_calculator, split, log=True):

    model.eval()

    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)
    ref_embeddings, ref_labels = get_all_embeddings(ref_set, model)
    ref_labels = ref_labels.squeeze(1)

    print("Computing accuracy...")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
        ref_embeddings,
        test_labels,
        ref_labels,
        embeddings_come_from_same_source=False)

    prec_1 = accuracies["precision_at_1"]
    if log:
        wandb.log({f"{split}/accuracy": prec_1})
    print(f"Accuracy on {split} set (Precision@1) = {prec_1}")

    return prec_1


def tester_ffnn(model, val_dataset, val_loader, device, split):

    model.eval()

    corr_preds = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = logits.argmax(-1)
            corr_preds += torch.sum(predictions == labels).item()

    acc = corr_preds / len(val_dataset)
    wandb.log({f"{split}/accuracy": acc})
    print(f"{split} set accuracy = {acc}")

    return acc


def trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=None, diff_sizes=False):

    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):

        labels = labels.to(device)
        data = [datum.to(device) for datum in data] if diff_sizes else data.to(device)
        optimizer.zero_grad()

        if diff_sizes:
            out_emb = []
            for datum in data:
                emb = model(datum.unsqueeze(0)).squeeze(0)
                out_emb.append(emb)
            embeddings = torch.stack(out_emb, dim=0)
        else:
            embeddings = model(data)

        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))


def trainer_ffnn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=False, diff_sizes=False):

    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        labels = labels.to(device)
        inputs = inputs.to(device)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss": loss.item()})
        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))

                    
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir_path", type=str, required=True,
        help="Root image directory path, with character class subfolders")
    parser.add_argument("--train_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was trained on")
    parser.add_argument("--val_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was validated on")
    parser.add_argument("--test_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was tested on")
    parser.add_argument("--run_name", type=str, required=True,
        help="Name of run for W&B logging purposes")
    parser.add_argument('--batch_size', type=int, default=128,
        help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-6,
        help="LR for AdamW")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
        help="Weight decay for AdamW")
    parser.add_argument('--num_epochs', type=int, default=5,
        help="Number of epochs")
    parser.add_argument('--temp', type=float, default=0.1,
        help="Temperature for InfoNCE loss")
    parser.add_argument('--start_epoch', type=int, default=1,
        help="Starting epoch")
    parser.add_argument('--m', type=int, default=4,
        help="m for m in m-class sampling")
    parser.add_argument('--imsize', type=int, default=224,
        help="Size of image for encoder")
    parser.add_argument("--hns_txt_path", type=str, default=None,
        help="Path to text file of mined hard negatives")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Load checkpoint before training")
    parser.add_argument("--lang", type=str, default="jp", choices=["jp", "en"],
        help="Language of characters being recognized")
    parser.add_argument('--finetune', action='store_true', default=False,
        help="Train just on target character crops")
    parser.add_argument('--pretrain', action='store_true', default=False,
        help="Train just on render character crops")
    parser.add_argument('--high_blur', action='store_true', default=False,
        help="Increase intensity of the blurring data augmentation for renders")
    parser.add_argument('--diff_sizes', action='store_true', default=False,
        help="DEPRECATED: allow different sizes for training crops")
    parser.add_argument('--epoch_viz_dir', type=str, default=None,
        help="Visualize and save some training samples by batch to this directory")
    parser.add_argument('--infer_hardneg_k', type=int, default=8,
        help="Infer k-NN hard negatives for each training sample, and save to a text file")
    parser.add_argument('--N_classes', type=int, default=None,
        help="Triggers use of FFNN classifier head with N classes")
    parser.add_argument('--test_at_end', action='store_true', default=False,
        help="Inference on test set at end of training with best val checkpoint")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument("--num_passes", type=int, default=1,
        help="Defines epoch as number of passes of N_chars * M")
    parser.add_argument('--no_aug', action='store_true', default=False,
        help="Turn off data augmentation")
    args = parser.parse_args()

    # setup

    wandb.init(project="effocr_recog_v2", name=args.run_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.run_name, exist_ok=True)
    with open(os.path.join(args.run_name, "args_log.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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

    # init encoder

    if args.checkpoint is None and args.N_classes is None:
        if not args.auto_model_timm is None:
            enc = encoder(args.auto_model_timm)
        elif not args.auto_model_hf is None:
            enc = encoder(args.auto_model_hf)
        else:
            enc = encoder()
    elif args.checkpoint is None and not args.N_classes is None:
        if not args.auto_model_timm is None:
            enc = encoder(args.auto_model_timm)
        elif not args.auto_model_hf is None:
            enc = encoder(args.auto_model_hf)
        else:
            enc = encoder(n_classes=args.N_classes)
    elif not args.checkpoint is None and not args.N_classes is None:
        enc = encoder.load(args.checkpoint, n_classes=args.N_classes)
    else:
        enc = encoder.load(args.checkpoint)

    # data parallelism

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        datapara = True
        enc = nn.DataParallel(enc)
    else:
        datapara = False
    
    # create dataset

    train_dataset, val_dataset, test_dataset, \
        train_loader, val_loader, test_loader = create_dataset(
            args.root_dir_path, 
            args.train_ann_path,
            args.val_ann_path, 
            args.test_ann_path, 
            args.batch_size,
            hardmined_txt=args.hns_txt_path, 
            m=args.m, 
            finetune=args.finetune,
            pretrain=args.pretrain,
            high_blur=args.high_blur,
            lang=args.lang,
            knn=args.N_classes is None,
            diff_sizes=args.diff_sizes,
            imsize=args.imsize,
            num_passes=args.num_passes,
            no_aug=args.no_aug
        )

    render_dataset = create_render_dataset(
        args.root_dir_path,
        lang=args.lang,
        font_name="NotoSerifCJKjp-Regular" if args.lang == "jp" else "NotoSerif-Regular",
        imsize=args.imsize
    )
    
    # optimizer and loss

    optimizer = AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = losses.SupConLoss(temperature = args.temp) if args.N_classes is None else CrossEntropyLoss()

    # set tester

    if args.N_classes is None: # kNN classification
        tester = tester_knn
        accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
    else:                      # FFNN classification
        tester = tester_ffnn
        idx_to_class = {v: chr(int(k)) for k, v in val_dataset.class_to_idx.items()}
        with open(os.path.join(args.run_name, "class_map.json"), "w") as f:
            json.dump(idx_to_class, f, indent=2)
        assert len(idx_to_class.keys()) == args.N_classes, \
            f"WARNING: specified number of classes {args.N_classes} disagrees with number of classes in dataset {len(idx_to_class.keys())}"
    
    # get zero-shot accuracy

    print("Zero-shot accuracy:")
    if args.N_classes is None:
        best_acc = tester(val_dataset, render_dataset, enc, accuracy_calculator, "val", log=False)
    else:
        best_acc = tester(enc, val_dataset, val_loader, device, "val")

    # set trainer

    trainer = trainer_knn if args.N_classes is None else trainer_ffnn # kNN vs. FFNN

    # warm start training

    print("Training...")
    if not args.epoch_viz_dir is None: os.makedirs(args.epoch_viz_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.num_epochs+args.start_epoch):
        trainer(enc, loss_func, device, train_loader, optimizer, epoch, args.epoch_viz_dir, args.diff_sizes)
        if args.N_classes is None:
            acc = tester(val_dataset, render_dataset, enc, accuracy_calculator, "val")
        else:
            acc = tester(enc, val_dataset, val_loader, device, "val")
        if acc >= best_acc:
            best_acc = acc
            save_model(args.run_name, enc, "best", datapara)
        # scheduler.step()

    # save index

    del enc
    if args.N_classes is None:
        best_enc = encoder.load(os.path.join(args.run_name, "enc_best.pth"))
        save_ref_index(render_dataset, best_enc, args.run_name)
    else:
        best_enc = encoder.load(os.path.join(args.run_name, "enc_best.pth"))

    # optionally test at end...

    if args.test_at_end:
        if args.N_classes is None:
            test_acc = tester(test_dataset, render_dataset, best_enc, accuracy_calculator, "test")
            print(f"Final test acc: {test_acc}")
        else:
            test_acc = tester(best_enc, test_dataset, test_loader, device, "test")

    # optionally infer hard negatives (turned on by default, highly recommend to facilitate hard negative training)

    if not args.infer_hardneg_k is None and args.N_classes is None:
        query_paths = [x[0] for x in train_dataset.data if os.path.basename(x[0]).startswith("PAIRED")]
        if len(query_paths) == 0:
            print("No explicit training data... constructing hard neg from (unique) synth crops!")
            query_path_char_map = defaultdict(list)
            query_paths = []
            for x in train_dataset.data:
                query_path_char_map[os.path.basename(x[0]).split("_")[0]].append(x[0])
            for k, v in query_path_char_map.items():
                query_paths.append(np.random.choice(v))
        print(f"Num hard neg paths: {len(query_paths)}")
        transform = create_paired_transform(args.imsize)
        infer_hardneg(query_paths, train_dataset if args.finetune else render_dataset, best_enc, 
            os.path.join(args.run_name, "ref.index"), 
            transform, os.path.join(args.run_name, "hns.txt"), 
            k=args.infer_hardneg_k, finetune=args.finetune)

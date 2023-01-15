#!/usr/bin/env python

# adapted from detectron2/tools/lazyconfig_train_net.py 
# @ https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py

import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
import omegaconf

logger = logging.getLogger("detectron2")
import wandb
import pprint
import copy
import json
import os


def do_test(cfg, model):

    if "evaluator" in cfg.dataloader:

        ret = inference_on_dataset(
            model, 
            instantiate(cfg.dataloader.test), 
            instantiate(cfg.dataloader.evaluator)
        )

        print_csv_format(ret)
        return ret


def do_train(args, cfg):

    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """

    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) if comm.is_main_process() else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.BestCheckpointer(cfg.train.eval_period, checkpointer, "bbox/AP"),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            ) if comm.is_main_process() else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)

    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    trainer.train(start_iter, cfg.train.max_iter)


def register_dataset(root, name, images, train_file, val_file, test_file):

    image_dir = f"{root}/{images}"
    register_coco_instances(f"{name}_train", {}, f"{root}/{train_file}", image_dir)
    register_coco_instances(f"{name}_val",   {}, f"{root}/{val_file}",   image_dir)
    register_coco_instances(f"{name}_test",  {}, f"{root}/{test_file}",  image_dir)


def main(args):

    # setup wandb proj

    wandb.setup()
    if comm.is_main_process():
        wandb.init(project="EffOCR Localizer D2", name=args.name, sync_tensorboard=True)

    # register dataset

    register_dataset(
        root=args.dataset_root,
        images=args.images_dir,
        name=args.dataset_name,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file
    )

    # set hyperparameters and model config

    cfg = LazyConfig.load(args.config_file)

    cfg.dataloader.train.dataset.names = f"{args.dataset_name}_train"
    cfg.dataloader.train.total_batch_size = args.batch_size

    # approx. match data augmentations of MMDetection, not necessary

    if args.alt_augs:
        cfg.dataloader.train.mapper.augmentations = [
            {
                "horizontal": True,
                "_target_": "RandomFlip"
            },
            {
                "short_edge_length": [480,512,544,576,608,640,672,704,736,768,800],
                "max_size": 1333,
                "sample_style": "choice",       
                "_target_": "ResizeShortestEdge"
            },
        ]

        cfg.dataloader.test.mapper.augmentations = [
            {
                "short_edge_length": 800,
                "max_size": 1333,
                "_target_": "ResizeShortestEdge"
            },
        ]

    # other config settings

    cfg.dataloader.test.dataset.names = f"{args.dataset_name}_val"

    cfg.dataloader.eval = copy.deepcopy(cfg.dataloader.test)
    cfg.dataloader.eval.dataset.names = f"{args.dataset_name}_test"

    cfg.dataloader.evaluator.dataset_name = f"{args.dataset_name}_val"

    cfg.train.checkpointer.max_to_keep = 10
    cfg.train.checkpointer.period = 1000
    cfg.train.log_period = 100

    cfg.train.eval_period = 1000

    dataset = DatasetCatalog.get(cfg.dataloader.train.dataset.names)
    max_iter = len(dataset) * args.num_epochs
    print(f"Max iter: {max_iter}")
    cfg.train.max_iter = max_iter // cfg.dataloader.train.total_batch_size

    cfg.model.roi_heads.num_classes = 2
    cfg.model.roi_heads.mask_head.num_classes = 2
    # cfg.model.roi_heads.box_predictor.num_classes = 2

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.train.output_dir = args.output_dir
    if not args.init_ckpt is None:
        cfg.train.init_checkpoint = args.init_ckpt

    anchors = [[x*2, x*8, x*32] for x in [4, 8, 16, 32, 64]]
    cfg.model.proposal_generator.anchor_generator.sizes = anchors
    cfg.model.proposal_generator.head.num_anchors = len(anchors[0]) * 3

    if not args.lr is None:
        cfg.optimizer.lr = args.lr

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    pp = pprint.PrettyPrinter(indent=2)
    config_as_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    pp.pprint(config_as_dict)
    with open(os.path.join(args.output_dir, "omega_config.json"), "w") as f:
        json.dump(config_as_dict, f, indent=2, default=lambda o: getattr(o, '__name__', None))

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":

    parser = default_argument_parser()
    parser.add_argument("--name", type=str, required=True,
        help="Run name for W&B logging")
    parser.add_argument("--init_ckpt", type=str, default=None,
        help="Checkpoint for initializing model, defaults to D2 recommendation")
    parser.add_argument("--lr", type=float, default=None,
        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=40,
        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size")
    parser.add_argument('--alt_augs', action='store_true', default=False,
        help="Approximately copy one version of mmdetection's data augmentation strategy, not recommended")
    parser.add_argument("--dataset_root", type=str, required=True,
        help="Root dir of training dataset in COCO format")
    parser.add_argument("--dataset_name", type=str,required=True,
        help="Name to give dataset for registration")
    parser.add_argument("--train_file", type=str, required=True,
        help="Name of COCO JSON file with training data (not full path)")
    parser.add_argument("--val_file", type=str, required=True,
        help="Name of COCO JSON file with validation data (not full path)")
    parser.add_argument("--test_file", type=str, required=True,
        help="Name of COCO JSON file with test data (not full path)")
    parser.add_argument("--output_dir", type=str, required=True,
        help="Directory for outputs of training process")
    parser.add_argument("--images_dir", type=str, required=True,
        help="Directory for outputs of training process")
    args = parser.parse_args()

    wandb.require("service")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

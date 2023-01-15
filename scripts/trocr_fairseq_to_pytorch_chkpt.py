# adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/trocr/convert_trocr_unilm_to_pytorch.py
"""Convert TrOCR checkpoints from the unilm repository."""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import (
    TrOCRConfig,
    TrOCRForCausalLM,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    # cls token, position embeddings and patch embeddings of encoder
    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, encoder_config):
    for i in range(encoder_config.num_hidden_layers):
        # queries, keys and values (only weights, no biases)
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisionEncoderDecoderModel structure.
    """
    # define encoder and decoder configs based on checkpoint_url
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    decoder_config = TrOCRConfig()

    # size of the architecture
    trocr_base_config = {
        "architectures": [
            "VisionEncoderDecoderModel"
        ],
        "decoder": {
            "_name_or_path": "",
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "add_cross_attention": True,
            "architectures": None,
            "attention_dropout": 0.0,
            "bad_words_ids": None,
            "bos_token_id": 0,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": 0.0,
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 12,
            "decoder_start_token_id": 2,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "dropout": 0.1,
            "early_stopping": False,
            "cross_attention_hidden_size": 768,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": 2,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layernorm_embedding": False,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 1024,
            "min_length": 0,
            "model_type": "trocr",
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 1,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "scale_embedding": True,
            "sep_token_id": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": False,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.12.0.dev0",
            "use_bfloat16": False,
            "use_cache": False,
            "use_learned_position_embeddings": False,
            "vocab_size": 50265
        },
        "encoder": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "cross_attention_hidden_size": None,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "image_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "min_length": 0,
            "model_type": "vit",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 12,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "patch_size": 16,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "qkv_bias": False,
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.12.0.dev0",
            "use_bfloat16": False
        },
        "is_encoder_decoder": True,
        "model_type": "vision-encoder-decoder",
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": None
    }

    trocr_small_config = {
        "architectures": [
            "VisionEncoderDecoderModel"
        ],
        "decoder": {
            "_name_or_path": "",
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "add_cross_attention": True,
            "architectures": None,
            "attention_dropout": 0.0,
            "bad_words_ids": None,
            "bos_token_id": 0,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": 0.0,
            "cross_attention_hidden_size": 384,
            "d_model": 256,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 1024,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 6,
            "decoder_start_token_id": 2,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "dropout": 0.1,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": 2,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layernorm_embedding": True,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 512,
            "min_length": 0,
            "model_type": "trocr",
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 1,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "scale_embedding": True,
            "sep_token_id": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": False,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.14.1",
            "use_bfloat16": False,
            "use_cache": False,
            "use_learned_position_embeddings": True,
            "vocab_size": 64044
        },
        "encoder": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 384,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "image_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 1536,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "min_length": 0,
            "model_type": "deit",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 6,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "patch_size": 16,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "qkv_bias": True,
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.14.1",
            "use_bfloat16": False
        },
        "eos_token_id": 2,
        "is_encoder_decoder": True,
        "model_type": "vision-encoder-decoder",
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": None
    }

    config_of_interest = trocr_base_config if args.model == "base" else trocr_small_config

    for k, v in config_of_interest["encoder"].items():
        if hasattr(encoder_config, k):
            setattr(encoder_config, k, v)
    for k, v in config_of_interest["decoder"].items():
        if hasattr(decoder_config, k):
            setattr(decoder_config, k, v)

    # load HuggingFace model
    encoder = ViTModel(encoder_config, add_pooling_layer=False)
    decoder = TrOCRForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # load state_dict of original model, rename some keys
    state_dict = torch.load(checkpoint_url, map_location="cpu")["model"]

    rename_keys = create_rename_keys(encoder_config, decoder_config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, encoder_config)

    # remove parameters we don't need
    del state_dict["encoder.deit.head.weight"]
    del state_dict["encoder.deit.head.bias"]
    del state_dict["decoder.version"]
    del state_dict["encoder.embeddings.position_embeddings"]

    # add prefix to decoder keys
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val
        else:
            state_dict[key] = val

    # load state dict
    model.load_state_dict(state_dict, strict=False)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--model", default="base", type=str, help="Size of model being converted."
    )
    args = parser.parse_args()
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
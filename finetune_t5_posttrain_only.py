#!/usr/bin/env python3
import argparse
import os
import datasets
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import utils

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

MAX_GPU_SAMPLES = int(os.environ.get("MAX_GPU_SAMPLES"))

logger_fields = {
    "training_loss": [],
    "validation": {},
    "test": [],
    "metadata": {},
    "errors": []
}


def main(config):
    logger = utils.simple_logger(config)

    if os.path.isdir(os.path.join(os.path.abspath(os.getcwd()), config.model_path)):
        print(f"loading model from {config.model_path}")
        model = T5ForConditionalGeneration.from_pretrained(config.model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    xsum = datasets.load_dataset('xsum')
    validation_dataset = xsum['validation']

    if config.validate:
        validation_set = utils.XSUM_dataset(validation_dataset, tokenizer)
        validation_loader = DataLoader(
            validation_set, batch_size=MAX_GPU_SAMPLES*8, shuffle=False
        )
        utils.validate("post", model, device, validation_loader,
                       tokenizer, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune t5 on extreme summarization")
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=MAX_GPU_SAMPLES, help="batch size")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=1e-6, help="learning rate")
    parser.add_argument('--log_path', type=str, default='testing.json')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('-fp16', action='store_true')
    parser.add_argument('-v', '--validate', action='store_true')

    args = parser.parse_args()
    assert(args.batch_size % MAX_GPU_SAMPLES ==
           0), f"Batch size ({args.batch_size}) must be a multiple of {MAX_GPU_SAMPLES}"

    main(args)

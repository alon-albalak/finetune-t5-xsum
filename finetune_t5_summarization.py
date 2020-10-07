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
    train_dataset = xsum['train']
    validation_dataset = xsum['validation']
    test_dataset = xsum['test']

    train_set = utils.XSUM_dataset(train_dataset, tokenizer)
    training_loader = DataLoader(
        train_set, batch_size=MAX_GPU_SAMPLES, shuffle=True
    )
    if config.validate:
        validation_set = utils.XSUM_dataset(validation_dataset, tokenizer)
        validation_loader = DataLoader(
            validation_set, batch_size=MAX_GPU_SAMPLES*8, shuffle=False
        )
        utils.validate("pre", model, device, validation_loader,
                       tokenizer, logger)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate)

    gradient_accumulation_steps = config.batch_size/MAX_GPU_SAMPLES

    for epoch in range(config.epochs):
        if config.fp16:
            utils.train_fp16(epoch, model, device,
                             training_loader, optimizer,
                             gradient_accumulation_steps,
                             logger, tokenizer)
        else:
            utils.train(epoch, model, device,
                        training_loader, optimizer,
                        gradient_accumulation_steps,
                        logger, tokenizer)
        if config.model_path:
            model.save_pretrained(config.model_path)

    if config.validate:
        validation_set = utils.XSUM_dataset(validation_dataset, tokenizer)
        validation_loader = DataLoader(
            validation_set, batch_size=MAX_GPU_SAMPLES*4, shuffle=False
        )
        utils.validate("post", model, device, validation_loader,
                       tokenizer, logger)

    test_set = utils.XSUM_dataset(test_dataset, tokenizer)
    test_loader = DataLoader(
        test_set, batch_size=MAX_GPU_SAMPLES*8, shuffle=False)
    utils.test(model, device, test_loader, tokenizer, logger)


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

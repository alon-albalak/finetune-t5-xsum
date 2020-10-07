import json
import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
import datasets
from bert_score import score as bertscore
from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_SOURCE_LEN = 512
MAX_SUMMARY_LEN = 150
DECODING_BEAMS = 2


class simple_logger():
    def __init__(self, config):
        self.save_path = os.path.join(
            os.path.abspath(os.getcwd()), config.log_path)

        if os.path.isfile(self.save_path):
            print(f"Loading log from {self.save_path}")
            self.logger = json.load(open(self.save_path))

        else:
            print(f"Initializing log at {self.save_path}")
            self.logger = {
                "training_loss": [],
                "validation": {},
                "test": [],
                "metadata": {},
                "errors": []
            }
            self.logger['metadata'] = {arg: getattr(
                config, arg) for arg in vars(config)}
            self.save()

    def save(self):
        with open(self.save_path, "w") as p:
            json.dump(self.logger, p, indent=2)
        print(f"Saved log at {self.save_path}")


class XSUM_dataset(Dataset):
    def __init__(self, data, tokenizer, max_source_len=MAX_SOURCE_LEN, max_summary_len=MAX_SUMMARY_LEN):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_len = max_source_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        doc = self.data[index]['document']
        summary = self.data[index]['summary']
        inputs = self.tokenizer.batch_encode_plus([f"summarize: {doc}"],
                                                  max_length=self.max_source_len, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer.batch_encode_plus([summary],
                                                   max_length=self.max_summary_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        input_mask = inputs['attention_mask'].squeeze()
        target_ids = targets['input_ids'].squeeze()
        target_mask = targets['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "document": doc
        }


def train(epoch, model, device, data_loader, optimizer,
          gradient_accumulation_steps, logger, tokenizer=None):

    experiment_id = logger.logger['metadata']['model_path']
    bleu_metric = datasets.load_metric('bleu', experiment_id=experiment_id)
    rouge_metric = datasets.load_metric('rouge', experiment_id=experiment_id)
    bertscore_metric = bertscore

    model.train()
    for i, data in enumerate(tqdm(data_loader), 1):
        y = data['target_ids'].to(device)

        x_ids = data['input_ids'].to(device)
        x_mask = data['input_mask'].to(device)

        outputs = model(input_ids=x_ids, attention_mask=x_mask,
                        labels=y)

        loss = outputs[0]

        if (i) % 10 == 0:
            logger.logger['training_loss'].append([f"epoch {i}", loss.item()])

            if (i) % 500 == 0:
                logger.logger['training_loss'][-1].append(evaluate_batch(
                    model, y, x_ids, x_mask, tokenizer, data['document'], logger, bleu_metric, rouge_metric, bertscore_metric))
                logger.save()
                model.train()
                print(f"Epoch: {i}, Loss: {loss.item()}")

        loss.backward()

        if (i) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


# def train_fp16(epoch, model, device, data_loader, optimizer,
#                gradient_accumulation_steps, logger=None, tokenizer=None):

#     scaler = GradScaler()
#     model.train()

#     for i, data in enumerate(tqdm(data_loader), 1):
#         y = data['target_ids'].to(device)

#         x_ids = data['input_ids'].to(device)
#         x_mask = data['input_mask'].to(device)

#         with autocast():
#             outputs = model(input_ids=x_ids, attention_mask=x_mask,
#                             labels=y)
#             loss = outputs[0]

#         if (i) % 10 == 0 and logger:
#             logger.logger['training_loss'].append([f"epoch {i}", loss.item()])

#         if (i) % 500 == 0:
#             print(f"Epoch: {i}, Loss: {loss.item()}")
#             if logger:
#                 logger.save()

#         scaler.scale(loss).backward()

#         if (i) % gradient_accumulation_steps == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()


def validate(label, model, device, data_loader, tokenizer, logger):

    experiment_id = logger.logger['metadata']['model_path']
    bleu_metric = datasets.load_metric('bleu', experiment_id=experiment_id)
    rouge_metric = datasets.load_metric('rouge', experiment_id=experiment_id)
    bertscore_metric = bertscore

    logger.logger['validation'][label] = []

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader), 1):
            y = data['target_ids'].to(device)

            x_ids = data['input_ids'].to(device)
            x_mask = data['input_mask'].to(device)

            batch_evaluation = evaluate_batch(
                model, y, x_ids, x_mask, tokenizer, data['document'], logger, bleu_metric, rouge_metric, bertscore_metric)

            logger.logger['validation'][label].extend(batch_evaluation)

            if i % 10 == 0:
                logger.save()
    logger.save()


def test(model, device, data_loader, tokenizer, logger):
    experiment_id = logger.logger['metadata']['model_path']
    bleu_metric = datasets.load_metric('bleu', experiment_id=experiment_id)
    rouge_metric = datasets.load_metric('rouge', experiment_id=experiment_id)
    bertscore_metric = bertscore

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader), 1):
            y = data['target_ids'].to(device)

            x_ids = data['input_ids'].to(device)
            x_mask = data['input_mask'].to(device)

            batch_evaluation = evaluate_batch(
                model, y, x_ids, x_mask, tokenizer, data['document'], logger, bleu_metric, rouge_metric, bertscore_metric)

            logger.logger['test'].extend(batch_evaluation)

            if i % 10 == 0:
                logger.save()
    logger.save()


def evaluate_batch(model, y, x_ids, x_mask, tokenizer, docs, logger, bleu, rouge, bertscore):
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=x_ids,
            attention_mask=x_mask,
            max_length=MAX_SUMMARY_LEN,
            num_beams=DECODING_BEAMS,
            repetition_penalty=2.5,
            length_penalty=1,
            early_stopping=True
        )

    preds = [tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    targets = [tokenizer.decode(
        t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

    b_refs = [[t.split()] for t in docs]
    b_preds = [p.split() for p in preds]

    r_refs = targets
    r_preds = preds

    b_score = []
    r_score = []
    bs_score = "none"
    for ref, pred in zip(b_refs, b_preds):
        try:
            b_score.append(bleu.compute(
                predictions=[pred], references=[ref]))
        except ZeroDivisionError:
            logger.logger['errors'].append(
                ["BLEU Error", "Divide by zero", ref, pred])
            b_score.append({"Error": "Divide by zero"})
            continue
        except Exception as e:
            logger.logger['errors'].append(
                ["BLEU Error", "other", ref, pred])
            b_score.append({"Error": str(e)})
            continue

    for ref, pred in zip(r_refs, r_preds):
        r_score.append(rouge.compute(
            predictions=[pred], references=[ref]))

    bs_score = bertscore(cands=r_preds,
                         refs=r_refs, lang='en')
    (P, R, F1) = bs_score

    # metadata should be organized by sample: document, summary, reference_summary, scores, etc.
    batch = [
        {
            "document": doc,
            "summary": pred,
            "reference_summary": target,
            'bleu': {field: bl[field] for field in bl},
            'rouge': {field: rg[field] for field in rg},
            "bert_precision": p.item(),
            "bert_recall": r.item(),
            "bert_F1": f1.item()
        }
        for doc, pred, target, bl, rg, p, r, f1 in zip(docs, preds, targets, b_score, r_score, P, R, F1)
    ]

    return batch


def plot_training_loss(log_files):
    for l in log_files:
        tmp = json.load(open(l))
        base_label = l.split("/")[-1].split(".")[0]
        plt.plot([t[1] for t in tmp['training_loss']], label=base_label)
        plt.plot(running_mean([t[1]
                               for t in tmp['training_loss']], 50), label=base_label+" smoothed")

    plt.legend()
    plt.show()


def plot_metrics(log_files):
    metric_names = ['bleu',
                    'bleu_precision',
                    'rouge1',
                    'rougeL',
                    'bertscore_precision',
                    'bertscore_recall',
                    'bertscore_f1']

    all_metrics = {}
    for l in log_files:
        tmp = json.load(open(l))
        metrics = {
            'bleu': [],
            'bleu_precision': [],
            'rouge1': [],
            'rougeL': [],
            'bertscore_precision': [],
            'bertscore_recall': [],
            'bertscore_f1': []
        }
        for t in tmp['training_loss']:
            if len(t) > 2:
                # Add all metrics to metrics
                for i in range(len(t[2])):
                    if 'bleu' in t[2][i]['bleu'].keys():
                        metrics['bleu'].append(t[2][i]['bleu']['bleu'])
                        metrics['bleu_precision'].append(
                            t[2][i]['bleu']['precisions'][0])
                    metrics['rouge1'].append(t[2][i]['rouge']['rouge1'][0][0])
                    metrics['rougeL'].append(t[2][i]['rouge']['rougeL'][0][0])
                    metrics['bertscore_precision'].append(
                        t[2][i]['bert_precision'])
                    metrics['bertscore_recall'].append(t[2][i]['bert_recall'])
                    metrics['bertscore_f1'].append(t[2][i]['bert_F1'])
        all_metrics[l] = metrics

    for m in metric_names:
        for log_file in all_metrics:
            base_label = log_file.split("/")[-1].split(".")[0]
            plt.plot(all_metrics[log_file][m], label=base_label, alpha=0.5)
            plt.plot(running_mean(
                all_metrics[log_file][m], 10), label=base_label+" smoothed")

        plt.title(m)
        plt.legend()
        plt.show()


def plot_metrics_pre_post_training(log_files):
    metric_names = ['bleu',
                    'bleu_precision',
                    'rouge1',
                    'rougeL',
                    'bertscore_precision',
                    'bertscore_recall',
                    'bertscore_f1']

    pre = {l: {
        'bleu': [],
        'bleu_precision': [],
        'rouge1': [],
        'rougeL': [],
        'bertscore_precision': [],
        'bertscore_recall': [],
        'bertscore_f1': []
    } for l in log_files}
    post = copy.deepcopy(pre)

    for l in log_files:
        tmp = json.load(open(l))
        metrics = {
            'bleu': [],
            'bleu_precision': [],
            'rouge1': [],
            'rougeL': [],
            'bertscore_precision': [],
            'bertscore_recall': [],
            'bertscore_f1': []
        }
        for t in tmp['validation']['pre']:
            if 'bleu' in t['bleu'].keys():
                pre[l]['bleu'].append(t['bleu']['bleu'])
                pre[l]['bleu_precision'].append(t['bleu']['precisions'][0])
            pre[l]['rouge1'].append(t['rouge']['rouge1'][0][0])
            pre[l]['rougeL'].append(t['rouge']['rougeL'][0][0])
            pre[l]['bertscore_precision'].append(t['bert_precision'])
            pre[l]['bertscore_recall'].append(t['bert_recall'])
            pre[l]['bertscore_f1'].append(t['bert_F1'])

        for t in tmp['validation']['post']:
            if 'bleu' in t['bleu'].keys():
                post[l]['bleu'].append(t['bleu']['bleu'])
                post[l]['bleu_precision'].append(t['bleu']['precisions'][0])
            post[l]['rouge1'].append(t['rouge']['rouge1'][0][0])
            post[l]['rougeL'].append(t['rouge']['rougeL'][0][0])
            post[l]['bertscore_precision'].append(t['bert_precision'])
            post[l]['bertscore_recall'].append(t['bert_recall'])
            post[l]['bertscore_f1'].append(t['bert_F1'])

    pre_means = [[np.mean(pre[l][m]) for m in metric_names] for l in log_files]
    post_means = [[np.mean(post[l][m]) for m in metric_names]
                  for l in log_files]

    width = 0.2
    ind = list(range(len(metric_names)))
    plt.bar([i-(3*width/2) for i in ind], pre_means[0], width,
            label=f"{log_files[0].split('/')[-1]} pre")
    plt.bar([i-(width/2) for i in ind], post_means[0], width,
            label=f"{log_files[0].split('/')[-1]} post")

    plt.bar([i+(width/2) for i in ind], pre_means[1], width,
            label=f"{log_files[1].split('/')[-1]} pre")
    plt.bar([i+(3*width/2) for i in ind], post_means[1], width,
            label=f"{log_files[1].split('/')[-1]} post")

    plt.xticks([i+width/2 for i in ind], metric_names, rotation=45)
    plt.legend()
    plt.show()


def running_mean(x, N):
    new_x = x[:N-1]
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return np.concatenate((new_x, (cumsum[N:] - cumsum[:-N]) / float(N)))

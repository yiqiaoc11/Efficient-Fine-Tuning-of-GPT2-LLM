from typing import List, Tuple
import argparse
import torch
import transformers
import torch.nn as nn
import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
import icl
import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--k')
parser.add_argument('--mode', default='all')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()


DEVICE = torch.device(args.device)


class LoRAConv1DWrapper(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule
        self.shape = self.base_module.weight.shape
        self.A = nn.Linear(lora_rank, self.shape[1], bias=False)
        self.B = nn.Linear(self.shape[0], lora_rank, bias=False)
        nn.init.zeros_(self.B.weight)
        nn.init.normal_(self.A.weight)


    def forward(self, x):
        feat_weights = self.B(x)
        return self.base_module(x) + self.A(feat_weights)


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """

    num_transf = len(model.transformer.h)
    if mode == 'all':
        return model.parameters()
    elif mode == 'last':
        for i in range(num_transf-2):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        return model.parameters()
    elif mode == 'first':
        for i in range(2, num_transf):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        return model.parameters()
    elif mode == 'middle':
        for i in range(num_transf):
            for param in model.transformer.h[i].parameters():
                if i*2 != num_transf or (1+i)*2 != num_transf:
                    param.requires_grad = False
        return model.parameters()
    elif mode.startswith('lora'):
        for m in model.transformer.h:
            for param in m.parameters():
                param.requires_grad = False
            m.mlp.c_fc.A.weight.requires_grad = True
            m.mlp.c_proj.A.weight.requires_grad = True
            m.attn.c_attn.A.weight.requires_grad = True
            m.mlp.c_fc.B.weight.requires_grad = True
            m.mlp.c_proj.B.weight.requires_grad = True
            m.attn.c_attn.B.weight.requires_grad = True            
        return model.parameters()
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, you'll need to deal with the fact that different sequences witihn
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average loss is the *average loss over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """
    loss = nn.CrossEntropyLoss()
    if logits.dim() == 2:
        output = loss(logits, targets)
        return output
    elif logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        indices = (targets != -100).nonzero(as_tuple=True)[0]
        output = loss(logits[indices-1, :], targets[indices])
        return output
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')


def get_acc(logits, targets):
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, you'll need to deal with the fact that different sequences witihn
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
    """
    m = nn.LogSoftmax(dim=1)
    if logits.dim() == 2:
        return torch.mean((torch.argmax(logits, dim=1) == targets).float())
    elif logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        indices = (targets != -100).nonzero(as_tuple=True)[0]
        log_probs = m(logits[indices-1, :])
        return torch.mean((torch.argmax(log_probs, dim=1) == targets[indices]).float())
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')


def ft_bert(model, tok, x, y, mode, batch_size=8):
    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(x, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok([x[i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    return model


def tokenize_gpt2_batch(tokenizer, x, y):
    """
    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example
    
    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length] 
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length] 
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (you can get the tokenizer to handle this padding!).

    """
    tokenized_sequences = None
    tokenized_sequences = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True).to(DEVICE)
    
    max_len = tokenized_sequences['input_ids'].size()[1]
    tokenizer_x = torch.tensor([[len(s)] for s in tokenizer(x)['input_ids']])
    tokenizer_y = torch.tensor([[len(s)] for s in tokenized_sequences['input_ids']])
    base = torch.arange(max_len).expand_as(tokenized_sequences['input_ids'])
    lmask = base < tokenizer_x
    rmask = base >= tokenizer_y
    mask = lmask | rmask
    labels = tokenized_sequences['input_ids'].clone()
    labels[mask] = -100         # http://juditacs.github.io/2018/12/27/masked-attention.html
    tokenized_sequences['labels'] = labels
    return tokenized_sequences


def add_prefixes(x: List[str], y: List[str], dataset: str) -> Tuple[List[str], List[str]]:
    input_prefix = '' if utils.is_qa_dataset(dataset) else ''
    label_prefix = ' In the' if utils.is_qa_dataset(dataset) else ' TL;DR:'
    label_suffix = '.' if utils.is_qa_dataset(dataset) else ''

    x = [input_prefix + x_.replace('\n', ' ') + label_prefix for x_ in x]
    y = [' ' + y_.replace('\n', ' ') + label_suffix for y_ in y]

    return x, y


def ft_gpt2(model, tok, x, y, mode, dataset, batch_size=8, grad_accum=8):
    x, y = add_prefixes(x, y, dataset)

    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
    all_both = tokenize_gpt2_batch(tok, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[:batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum:]

        _x, _y = [x[i] for i in batch_idxs], [y[i] for i in batch_idxs]
        tokens_batch = tokenize_gpt2_batch(tok, _x, _y)
        logits = model(**tokens_batch, use_cache=False).logits
        loss = get_loss(logits, tokens_batch['labels'])
        loss = loss / grad_accum
        loss.backward()

        if step > 0 and step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx:idx+1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d['labels'])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')

            if total_acc >= utils.early_stop_thresold(dataset):
                print('Early stopping!')
                break
    return model


def eval(model, tok, val_data):
    x = tok(val_data['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    y = torch.tensor(val_data['y'], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(models: List[str], datasets: List[str], ks: List[int], modes: List[str], n_val: int = 125):
    results = {}
    for dataset in datasets:
        if args.debug:
            n_val = 1   
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)
        for model_name, mode in itertools.product(models, modes):
            if dataset == 'amazon':
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForSequenceClassification, num_labels=5)
            else:                
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
            stop_tokens = utils.stop_tokens(tokenizer)

            for k in ks:
                print(f'Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}')
                for repeat in range(args.repeats):
                    if repeat > 0:
                        print(f'Beginning repeat #{repeat}')
                    if dataset == 'amazon':
                        fine_tuned = ft_bert(model, tokenizer, train['x'][:k*5], train['y'][:k*5], mode)
                        val_acc = eval(fine_tuned, tokenizer, val)
                        results['_'.join([model_name, dataset, str(k), mode])] = val_acc
                    else:
                        if k > 0:
                            fine_tuned = ft_gpt2(model, tokenizer, train['x'][:k], train['simple_y'][:k], mode, dataset)
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))

                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt_mode = 'qa' if utils.is_qa_dataset(dataset) else 'tldr'
                            prompt = icl.get_icl_prompts([], [], test_input, prompt_mode=prompt_mode)
                            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
                            sampled_tokens = icl.do_sample(fine_tuned, input_ids, stop_tokens, max_tokens)
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = icl.get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')
                        results['_'.join([model_name, dataset, str(k), mode])] = metric

                    print(results)
                    question = 'ft'
                    if not os.path.exists(f'results/{question}'):
                        os.makedirs(f'results/{question}')

                    for k_, v in results.items():
                        with open(f'results/{question}/{k_}.json', 'w') as f:
                            json.dump({'metric': v}, f)
                    results = {}


def plot(models, datasets, ks, modes):
    data = defaultdict(lambda: defaultdict(list))
    question = 'ft'

    x_vals = set()
    for dataset in datasets:
        for model, mode in itertools.product(models, modes):
            for k in ks:
                fn = '_'.join([model, dataset, str(k), mode])
                id_ = '_'.join([model, dataset, mode])
                with open(f'results/{question}/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)

        for k, v in data.items():
            plt.plot(v['x'], v['y'], label=k)

    if max(x_vals) > 4:
        plt.xscale('symlog')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(' & '.join(datasets))
    plt.ylabel('/'.join([utils.metric_for_dataset(dataset) for dataset in datasets]))
    plt.xlabel('Number of support examples')
    # plt.show()
    plt.savefig('bAbI FET performance.png', bbox_inches="tight")

def run():
    ks = [int(k) for k in args.k.split(',')]
    if args.task == 'ft':
        run_ft(args.model.split(','), args.dataset.split(','), ks, args.mode.split(','))
    elif args.task == 'plot':
        plot(args.model.split(','), args.dataset.split(','), ks, args.mode.split(','))


if __name__ == '__main__':
    run()

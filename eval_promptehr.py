import os
import pdb

from torch.utils.data import DataLoader
import torch
import numpy as np

from promptehr.dataset import MimicDataset, MimicDataCollator
from promptehr.modeling_config import DataTokenizer, ModelTokenizer
from promptehr.model import BartForEHRSimulation
from promptehr.evaluator import Evaluator

# set visible device
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# avoid dead clock when taking multiple workers for dataloaders
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# initialize model & tokenizers (data & model)
data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')
input_dir = './data/processed'
data_tokenizer.extend_vocab(input_dir)
model_tokenizer = ModelTokenizer(data_tokenizer)

# let's build the model
model = BartForEHRSimulation.from_pretrained('./checkpoints/EHR-BART-all', model_tokenizer=model_tokenizer)
model.cuda()
model.eval()

mimic_dataset = MimicDataset(input_dir, 'test')
collate_fn = MimicDataCollator(data_tokenizer, mode='test', drop_feature=True)
evaluator = Evaluator(model, mimic_dataset, collate_fn, device=device)
code_types = ['diagnosis', 'procedure', 'drug']
ppl_types = ['tpl', 'spl']
for code_type in code_types:
    for ppl_type in ppl_types:
        ppl = evaluator.evaluate(code_type, ppl_type)
        print(f'code: {code_type}, ppl_type: {ppl_type}, value: {ppl}')

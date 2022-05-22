import os
import pdb
import json

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import TrainingArguments

from promptehr.dataset import MimicDataset, MimicDataCollator
from promptehr.modeling_config import EHRBartConfig, DataTokenizer, ModelTokenizer
from promptehr.trainer import PromptEHRTrainer
from promptehr.model import BartForEHRSimulation

# set visible device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# avoid dead clock when taking multiple workers for dataloaders
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# load data
input_dir = './data/processed'

# initialize model & tokenizers (data & model)
data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')
data_tokenizer.extend_vocab(input_dir)
model_tokenizer = ModelTokenizer(data_tokenizer)
configuration = EHRBartConfig(data_tokenizer, model_tokenizer, n_num_feature=1, cat_cardinalities=[2])

# build data collator and dataloader
mimic_dataset = MimicDataset(input_dir, mode='train')
mimic_val_dataset = MimicDataset(input_dir, mode='val')
mimic_train_collator = MimicDataCollator(data_tokenizer, max_train_batch_size=32, mode='train')
mimic_val_collator = MimicDataCollator(data_tokenizer, mode='val')

# let's build the model
model = BartForEHRSimulation(configuration, model_tokenizer)
model.cuda()

# let's build the trainer
training_args = TrainingArguments(
    per_device_train_batch_size=16, # no. of patients in each batch, default to be 1
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=1e-4,
    output_dir="./checkpoints/EHR-BART-all",
    num_train_epochs=50,
    save_steps=2000,
    eval_steps=2000,
    warmup_ratio=0.06,
    save_total_limit=10,
    logging_steps=100,
    dataloader_num_workers=12,
    dataloader_pin_memory=True,
    evaluation_strategy='steps',
    metric_for_best_model='eval_NLL_diagnosis',
    greater_is_better=False, # NLL is the less the better
    eval_accumulation_steps=10,
    load_best_model_at_end=True,
    logging_dir='./logs',      # directory for storing logs
    overwrite_output_dir=True,
)
trainer = PromptEHRTrainer(
    model=model,
    args=training_args,
    train_dataset=mimic_dataset,
    data_collator=mimic_train_collator,
    eval_dataset=mimic_val_dataset,
    val_data_collator=mimic_val_collator,
)
trainer.train()
trainer.save_model('./checkpoints/EHR-BART-all')
print('done')

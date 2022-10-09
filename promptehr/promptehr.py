'''
User interface to use promptEHR models.
'''

import os
import pdb
import json

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import TrainingArguments

from .dataset import MimicDataset, MimicDataCollator
from .modeling_config import EHRBartConfig, DataTokenizer, ModelTokenizer
from .trainer import PromptEHRTrainer
from .model import BartForEHRSimulation


class PromptEHR(nn.Module):
    '''
    Initialize a PromptEHR model to leverage language models to simulate sequential patient EHR data.

    Parameters:
    -----------
    token_dict: dict[list]
        A dictionary of new tokens (code events, e.g., ICD code) that the model needs to learn and generate.

    n_num_feature: int
        Number of numerical patient baseline features.

    cat_cardinalities: list[int]
        The number of categories for each categorical patient baseline features.

    epoch: int
        Num training epochs in total.

    batch_size: int
        Training batch size.

    eval_batch_size: int
        Evaluation batch size.
    
    eval_step: int
        How many steps of updates then try to evaluate the trained models.
    
    learning_rate: float
        Training learning rate.
    
    weight_decay: float
        Training weight decay.
    
    output_dir: str
        Training logs output to this folder.

    device: str or list[int]
        Should be str like `cuda:0` or `cpu`, otherwise should be a list GPU ids.
    '''
    def __init__(self,
        token_dict,
        n_num_feature,
        cat_cardinalities,
        epoch=50,
        batch_size=16,
        eval_batch_size=64,
        eval_step=1000,
        learning_rate=1e-5,
        weight_decay=1e-4,
        output_dir='./promptEHR_logs',
        device='cuda:0',
        ) -> None:
        super().__init__()
        self.data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')
        self.data_tokenizer.extend_vocab(token_dict)
        self.data_tokenizer.update_config(code_types=list(token_dict.keys()))
        self.config = {
            'token_dict': token_dict,
            'code_types': list(token_dict.keys()),
            'n_num_feature':n_num_feature,
            'cat_cardinalities':cat_cardinalities,
            'epoch':epoch,
            'batch_size':batch_size,
            'eval_batch_size':eval_batch_size,
            'eval_step':eval_step,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
        }
        self.model_tokenizer = ModelTokenizer(self.data_tokenizer)
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=n_num_feature, cat_cardinalities=cat_cardinalities)
        self.device = device
        if isinstance(device, list):
            self._set_visible_device(device)

        self._build_model()
        self.training_args = TrainingArguments(
            per_device_train_batch_size=batch_size, # no. of patients in each batch, default to be 1
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            output_dir=output_dir,
            num_train_epochs=epoch,
            save_steps=eval_step,
            eval_steps=eval_step,
            warmup_ratio=0.06,
            save_total_limit=10,
            logging_steps=100,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            evaluation_strategy='steps',
            metric_for_best_model='eval_ppl_diagnosis',
            greater_is_better=False, # NLL is the less the better
            eval_accumulation_steps=10,
            load_best_model_at_end=True,
            logging_dir=output_dir,      # directory for storing logs
            overwrite_output_dir=True,
            seed=123,
        )

        # avoid dead clock when taking multiple workers for dataloaders
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def fit(self, train_data, val_data=None):
        '''
        Fit PromptEHR model on the input training EHR data.

        Parameters
        ----------
        train_data: dict
            A dict of train dataset.

        val_data: dict
            A dict of valid dataset.
        '''
        mimic_dataset = train_data['dataset']
        mimic_val_dataset = val_data['dataset']
        
        mimic_train_collator = MimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_types'],
            max_train_batch_size=32, mode='train')
            
        mimic_val_collator = MimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_types'],
            mode='val')

        trainer = PromptEHRTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=mimic_dataset,
            data_collator=mimic_train_collator,
            eval_dataset=mimic_val_dataset,
            val_data_collator=mimic_val_collator,
            )
        trainer.train()

    def _build_model(self):
        self.model = BartForEHRSimulation(self.configuration, self.model_tokenizer)
        if isinstance(self.device, list): self.model.cuda()
        elif 'cuda' in self.device: self.model.cuda()
        else: self.model.cpu()

    def _set_visible_device(self, device):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device])
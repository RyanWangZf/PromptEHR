'''
User interface to use promptEHR models.
'''
import os
import pdb
import json
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import TrainingArguments
import numpy as np

from .dataset import MimicDataset, MimicDataCollator
from .modeling_config import EHRBartConfig, DataTokenizer, ModelTokenizer
from .trainer import PromptEHRTrainer
from .model import BartForEHRSimulation

class PromptEHR(nn.Module):
    '''
    Initialize a PromptEHR model to leverage language models to simulate sequential patient EHR data.

    Parameters:
    -----------
    code_type: list[str]
        A list of code types that the model will learn and generate.
        For example, `code_type=['diag','prod','med']`.

    token_dict: dict[list]
        A dictionary of new tokens (code events, e.g., ICD code) that the model needs to learn and generate.

    n_num_feature: int
        Number of numerical patient baseline features. Notice that it assumes that the input
        baseline features are `ALWAYS` numerical feature first. That is to say,
        the input baseline feature = [num1, num2, .., num_n, cat1, cat2,...].

    cat_cardinalities: list[int]
        The number of categories for each categorical patient baseline features.
        The input baseline feature = [num1, num2, .., num_n, cat1, cat2,...].

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
    
    num_worker: int
        Numer of dataloading paralleled processes.
    
    output_dir: str
        Training logs output to this folder.

    device: str or list[int]
        Should be str like `cuda:0` or `cpu`, otherwise should be a list GPU ids.
    '''
    def __init__(self,
        code_type,
        n_num_feature,
        cat_cardinalities,
        epoch=50,
        batch_size=16,
        eval_batch_size=64,
        eval_step=1000,
        learning_rate=1e-5,
        weight_decay=1e-4,
        num_worker=8,
        output_dir='./promptEHR_logs',
        device='cuda:0',
        ) -> None:
        super().__init__()
        self.data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')
        # will extend vocab after pass training data
        self.data_tokenizer.update_config(code_types=code_type)
        self.model_tokenizer = None
        self.config = {
            'code_type': code_type,
            'n_num_feature':n_num_feature,
            'cat_cardinalities':cat_cardinalities,
            'epoch':epoch,
            'batch_size':batch_size,
            'eval_batch_size':eval_batch_size,
            'eval_step':eval_step,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
        }
        # self.model_tokenizer = ModelTokenizer(self.data_tokenizer)
        # self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=n_num_feature, cat_cardinalities=cat_cardinalities)
        self.device = device

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
            dataloader_num_workers=num_worker, # debug
            dataloader_pin_memory=True,
            evaluation_strategy='steps',
            metric_for_best_model=f'eval_ppl_{code_type[0]}',
            greater_is_better=False, # NLL is the less the better
            eval_accumulation_steps=10,
            load_best_model_at_end=True,
            logging_dir=output_dir,      # directory for storing logs
            overwrite_output_dir=True,
            seed=123,
            no_cuda=True if self.device == 'cpu' else False, # if set CPU
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

        # create tokenizers based on the input data
        self._create_tokenizers(mimic_dataset)

        # can only build model after fit
        self._build_model()

        # start training
        self._fit(train_data=mimic_dataset,val_data=mimic_val_dataset)

    def _fit(self, train_data, val_data):
        mimic_train_collator = MimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            max_train_batch_size=32, mode='train')

        mimic_val_collator = MimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            mode='val')

        trainer = PromptEHRTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            data_collator=mimic_train_collator,
            eval_dataset=val_data,
            val_data_collator=mimic_val_collator,
            )
        trainer.train()
    
    def _create_tokenizers(self, train_data):
        # update data_tokenizer first
        def _collate_fn(inputs):
            outputs = defaultdict(list)
            for input in inputs:
                visit = input['v']
                for k,v in visit.items():
                    code_list = sum(v,[])
                    code_list = [k+'_'+str(c) for c in list(set(code_list))]
                    outputs[k].extend(code_list)
            return outputs
        dataloader = DataLoader(train_data, collate_fn=_collate_fn, batch_size=512, shuffle=False)
        for batch in dataloader:
            for k,v in batch.items():
                unq_codes = list(set(v))
                self.data_tokenizer.add_tokens(unq_codes)
                self.data_tokenizer.code_vocab[k] = np.array(unq_codes)
        
        self.model_tokenizer = ModelTokenizer(self.data_tokenizer)
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=self.config['n_num_feature'], cat_cardinalities=self.config['cat_cardinalities'])

    def _build_model(self):
        self.model = BartForEHRSimulation(self.configuration, self.model_tokenizer)

        if isinstance(self.device, list): 
            self._set_visible_device(self.device)
            self.model.cuda()
        elif 'cuda' in self.device: 
            self.model.cuda()
        else:
            # on cpu
            self._set_visible_device([])
            self.model.cpu()

    def _set_visible_device(self, device):
        if len(device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device])
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

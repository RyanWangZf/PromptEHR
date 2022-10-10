'''
User interface to use promptEHR models.
'''
import os
import pdb
import json
import math
import glob
from collections import defaultdict

import dill
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
    sample_config = {
        'num_beams': 1, # >1: beam_sample; =1: sample_gen
        'no_repeat_ngram_size': 1,
        'do_sample': True,
        'num_return_sequences': 1,
        'code_type': 'diagnosis',
        'top_k': 1,
        'temperature': 1.0,
        'max_length': 6,
    }
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
            save_total_limit=5,
            logging_steps=eval_step,
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
        train_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        val_data: dict
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.
        '''
        # create tokenizers based on the input data
        self._create_tokenizers(train_data)

        # can only build model after fit
        self._build_model()

        # start training
        self._fit(train_data=train_data,val_data=val_data)
    
    def predict(self, test_data, n_per_sample=None, n=None, sample_config=None):
        '''
        Generate synthetic records based on input real patient seq data.

        Parameters
        ----------
        test_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        n: int
            How many samples in total will be generated.

        n_per_sample: int
            How many samples generated based on each indivudals.

        sample_config: dict
            Configuration for sampling synthetic records, key parameters:
            'num_beams': Number of beams in beam search, if set `1` then beam search is deactivated;
            'top_k': Sampling from top k candidates.
            'temperature': temperature to make sampling distribution flater or skewer.        
        Returns
        -------
        Synthetic patient records in `SequencePatient` format.
        '''
        if n is not None: assert isinstance(n, int), 'Input `n` should be integer.'
        if n_per_sample is not None: assert isinstance(n_per_sample, int), 'Input `n_per_sample` should be integer.'
        n, n_per_sample = self._compute_n_per_sample(len(test_data), n, n_per_sample)

        if sample_config is not None:
            self.sample_config.update(sample_config)
            print('### Sampling Config ###')
            print(self.sample_config)

        # get test data loader
        test_dataloader = self._get_test_dataloader(test_data)

        # make generation
        outputs = self._predict_on_dataloader(test_dataloader, n, n_per_sample)

        pdb.set_trace()

        pass
    
    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.

        Parameters
        ----------
        output_dir: str
            The dir to save the learned model.
        '''
        make_dir_if_not_exist(output_dir)
        self._save_config(config=self.config, output_dir=output_dir)
        self._save_checkpoint(output_dir=output_dir)
        print('Save the trained model to:', output_dir)
    
    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        data_tokenizer_file, model_tokenizer_file = check_tokenizer_file(checkpoint)

        # load config
        self.config = self._load_config(config_filename)

        # load data tokenizer and model tokenizer
        self._load_tokenizer(data_tokenizer_file, model_tokenizer_file)

        # load configuration
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=self.config['n_num_feature'], cat_cardinalities=self.config['cat_cardinalities'])
        self.configuration.from_pretrained(checkpoint)

        # build model
        self._build_model()

        # load checkpoint
        state_dict = torch.load(checkpoint_filename, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        print('Load the pre-trained model from:', checkpoint)


    def _save_config(self, config, output_dir=None):        
        temp_path = os.path.join(output_dir, 'model_config.json')

        if os.path.exists(temp_path):
            os.remove(temp_path)

        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(config, indent=4)
            )

        # save the data tokenizer and model tokenizer of the model
        temp_path = os.path.join(output_dir, 'data_tokenizer.pkl')

        if os.path.exists(temp_path):
            os.remove(temp_path)

        with open(temp_path, 'wb') as f:
            dill.dump(self.data_tokenizer, f)

        temp_path = os.path.join(output_dir, 'model_tokenizer.pkl')

        if os.path.exists(temp_path):
            os.remove(temp_path)

        with open(temp_path, 'wb') as f:
            dill.dump(self.model_tokenizer, f)

        # save configuration
        self.configuration.save_pretrained(output_dir)

    def _load_tokenizer(self, data_tokenizer_file, model_tokenizer_file):
        with open(data_tokenizer_file, 'rb') as f:
            self.data_tokenizer = dill.load(f)

        with open(model_tokenizer_file, 'rb') as f:
            self.model_tokenizer = dill.load(f)

    def _load_config(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        return config

    def _get_test_dataloader(self, dataset):
        dataloader = DataLoader(dataset,
                batch_size=1, # one patient once
                drop_last=False,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                )
        return dataloader

    def _save_checkpoint(self,
                        epoch_id=0,
                        is_best=False,
                        output_dir=None,
                        filename='checkpoint.pth.tar'):

        if epoch_id < 1:
            filepath = os.path.join(output_dir, 'latest.' + filename)
        elif is_best:
            filepath = os.path.join(output_dir, 'best.' + filename)
        else:
            filepath = os.path.join(self.checkout_dir,
                                    str(epoch_id) + '.' + filename)
        
        # save statedict
        state_dict = self.state_dict()
        torch.save(state_dict, filepath)

    def _predict_on_dataloader(self, dataloader, n, n_per_sample):
        total_number = 0
        data_iterator = iter(dataloader)

        while total_number < n:
            try:
                data = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                data = next(data_iterator)

            pdb.set_trace()

        pass

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

    def _compute_n_per_sample(self, n_test_sample, n=None, n_per_sample=None):
        if n_per_sample is not None:
            n_total = n_test_sample*n_per_sample
            if n is not None:
                n_total = min(n_total, n)
            return n_total, n_per_sample
        else:
            return n, math.ceil(n_test_sample / n)

    def _get_num_visit(self, data, idx):
        visits = data['v']
        num_visit_list = []
        for k,v in visits.items():
            num_visit_list.append(len(v[idx]))
        
        num_visit_uq = list(set(num_visit_list))
        assert len(num_visit_uq) == 1, f'Find mismatch in the number of visit events {num_visit_list}, please check the input data {visits}.'
        return num_visit_uq[0]


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_checkpoint_file(input_dir, suffix='pth.tar'):
    '''
    Check whether the `input_path` is directory or to the checkpoint file.
        If it is a directory, find the only 'pth.tar' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.
    suffix: 'pth.tar' or 'model'
        The checkpoint file suffix;
        If 'pth.tar', the saved model is a torch model.
        If 'model', the saved model is a scikit-learn based model.
    '''
    suffix = '.' + suffix
    if input_dir.endswith(suffix):
        return input_dir

    ckpt_list = glob.glob(os.path.join(input_dir, '*'+suffix))
    assert len(ckpt_list) <= 1, f'Find more than one checkpoints under the dir {input_dir}, please specify the one to load.'
    assert len(ckpt_list) > 0, f'Do not find any checkpoint under the dir {input_dir}.'
    return ckpt_list[0]


def check_model_config_file(input_dir):
    '''
    Check whether the `input_path` is directory or to the `model_config.json` file.
        If it is a directory, find the only '.json' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.
    '''
    if input_dir.endswith('.json'):
        return input_dir

    if not os.path.isdir(input_dir):
        # if the input_dir is the given checkpoint model path,
        # we need to find the config file under the same dir.
        input_dir = os.path.dirname(input_dir)

    ckpt_list = glob.glob(os.path.join(input_dir, '*.json'))

    if len(ckpt_list) == 0:
        return None

    # find model_config.json under this input_dir
    model_config_name = [config for config in ckpt_list if 'model_config.json' in config]
    if len(model_config_name) == 1:
        return model_config_name[0]

    # if no model_config.json found, retrieve the only .json file.
    assert len(ckpt_list) <= 1, f'Find more than one config .json under the dir {input_dir}.'
    return ckpt_list[0]


def check_tokenizer_file(input_dir):
    return os.path.join(input_dir,'data_tokenizer.pkl'), os.path.join(input_dir,'model_tokenizer.pkl')

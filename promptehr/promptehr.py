'''
User interface to use promptEHR models.
'''
import os
import pdb
import json
import math
import glob
import random
import copy
from collections import defaultdict

import dill
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import TrainingArguments
import numpy as np
from tqdm import tqdm

from .dataset import MimicDataset, MimicDataCollator
from .modeling_config import EHRBartConfig, DataTokenizer, ModelTokenizer
from .trainer import PromptEHRTrainer
from .evaluator import Evaluator
from .model import BartForEHRSimulation
from . import constants

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

    n_num_feature: int (default=None)
        Number of numerical patient baseline features. Notice that it assumes that the input
        baseline features are `ALWAYS` numerical feature first. That is to say,
        the input baseline feature = [num1, num2, .., num_n, cat1, cat2,...].
        If not specified, the model will never include baseline features
        for conditional generation!

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
        code_type=None,
        n_num_feature=None,
        cat_cardinalities=None,
        epoch=50,
        batch_size=16,
        eval_batch_size=16,
        eval_step=1000,
        learning_rate=5e-5,
        weight_decay=1e-4,
        num_worker=8,
        output_dir='./promptEHR_logs',
        device='cuda:0',
        seed=123,
        ) -> None:
        super().__init__()
        self.data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')

        # will extend vocab after pass training data
        if code_type is not None:
            self.data_tokenizer.update_special_token_config(code_types=code_type)
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
        if isinstance(device, list):
            self._set_visible_device(device=device)
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
            max_grad_norm=0.5,
            save_total_limit=5,
            logging_steps=eval_step,
            dataloader_num_workers=num_worker, # debug
            dataloader_pin_memory=True,
            evaluation_strategy='steps',
            metric_for_best_model=f'eval_ppl_{code_type[0]}' if code_type is not None else None,
            greater_is_better=False, # NLL is the less the better
            eval_accumulation_steps=10,
            load_best_model_at_end=True,
            logging_dir=output_dir,      # directory for storing logs
            overwrite_output_dir=True,
            seed=seed,
            no_cuda=True if self.device == 'cpu' else False, # if set CPU
            )

        # avoid dead clock when taking multiple workers for dataloaders
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.model = None

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
    
    def predict(self, test_data, n_per_sample=None, n=None, sample_config=None, verbose=None):
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
        
        verbose: bool
            If print the progress bar or not.

        Returns
        -------
        Synthetic patient records in `SequencePatient` format.
        '''
        if n is not None: assert isinstance(n, int), 'Input `n` should be integer.'
        if n_per_sample is not None: assert isinstance(n_per_sample, int), 'Input `n_per_sample` should be integer.'
        assert (not n_per_sample is None) or (not n is None), 'Either `n` or `n_per_sample` should be provided to generate.'
        assert isinstance(self.model, BartForEHRSimulation), 'Model not found! Please fit the model or load the model from pretrained checkpoint first.'

        n, n_per_sample = self._compute_n_per_sample(len(test_data), n, n_per_sample)

        if sample_config is not None:
            self.sample_config.update(sample_config)
            print('### Sampling Config ###')
            print(self.sample_config)

        # get test data loader
        test_dataloader = self._get_test_dataloader(test_data)

        # make generation
        outputs = self._predict_on_dataloader(test_dataloader, n, n_per_sample, verbose=verbose)

        # formulate outputs to standard sequencepatient data
        # need 'visit', 'order', 'feature', 'n_num_feature', 'cat_cardinalties'
        visits, features, labels = [], [], []
        for output in outputs:
            code_types = [c for c in self.config['code_type'] if c in output]
            num_visit = len(output[code_types[0]])
            visit, feature = [], []
            for n in range(num_visit):
                visit_ = [output[code][n] for code in code_types]
                visit.append(visit_)
            visits.append(visit)
            if 'x_num' in output:
                feature.extend(output['x_num'])
            if 'x_cat' in output:
                feature.extend(output['x_cat'])
            if len(feature) > 0:
                features.append(feature)
            if 'y' in output: labels.append(output['y'])
        
        if len(features) > 0:
            features = np.stack(features, 0)
        else:
            features = None

        return_res = {
            'visit':visits, 
            'feature':features, 
            'order':self.config['code_type'],
            'n_num_feature':self.config['n_num_feature'],
            'cat_cardinalties':self.config['cat_cardinalities'],
            'y':labels,
            'voc': test_data.metadata['voc'],
        }
        return return_res

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

    def evaluate(self, test_data):
        '''
        Evaluate the trained PromptEHR model on the input data, will test the perplexity
        for each type of codes.
        
        Parameters
        ----------
        test_data: PatientSequence
            Standard sequential patient records in `PatientSequence` format.
        '''
        self.model.eval()
        self.eval()

        collator = MimicDataCollator(
            self.data_tokenizer,
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            mode='test', 
            drop_feature=False
            )

        evaluator = Evaluator(
            self.model,
            test_data,
            collator,
            device='cpu' if self.device == 'cpu' else 'cuda:0',
        )

        code_types = self.config['code_type']
        ppl_types = ['tpl','spl']
        for code_type in code_types:
            for ppl_type in ppl_types:
                ppl = evaluator.evaluate(code_type, ppl_type, eval_batch_size=self.config['eval_batch_size'])
                print(f'code: {code_type}, ppl_type: {ppl_type}, value: {ppl}')

    def from_pretrained(self, input_dir='./simulation/pretrained_promptEHR'):
        '''
        Load pretrained PromptEHR model and make patient EHRs generation.
        Pretrained model was learned from MIMIC-III patient sequence data.
        '''
        if input_dir is None:
            input_dir = './simulation/pretrained_promptEHR'
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)            
            url = constants.PRETRAINED_MODEL_URL
            download_pretrained(url, input_dir)
            print(f'Download pretrained PromptEHR model, save to {input_dir}.')
        
        print('Load pretrained PromptEHR model from', input_dir)
        self.load_model(input_dir)
    
    def update_config(self, config):
        '''
        Update the configuration of the model.

        Parameters
        ----------
        config: dict
            The configuration of the model.
            Refer to the `config` in `__init__` for more details.
        '''
        self.config.update(config)
        
        # update training args
        train_args = copy.deepcopy(config)
        for k, v in config.items():
            if k in constants.config_to_train_args:
                train_args[constants.config_to_train_args[k]] = v
                train_args.pop(k)
        
        for k,v in train_args.items():
            if hasattr(self.training_args, k):
                setattr(self.training_args, k, v)
        
        # important when you train the model with different datasets
        code_type = self.config['code_type']
        self.training_args.metric_for_best_model = \
            f'eval_ppl_{code_type[0]}' if code_type is not None else None,

        print('### Model Config ###')
        print(self.config)

        print('### Training Args ###')
        print(self.training_args)

    def _save_config(self, config, output_dir=None):        
        temp_path = os.path.join(output_dir, 'promptehr_config.json')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(config, indent=4)
            )

        # save the data tokenizer and model tokenizer of the model
        temp_path = os.path.join(output_dir, 'data_tokenizer.pkl')
        with open(temp_path, 'wb') as f:
            dill.dump(self.data_tokenizer, f)

        temp_path = os.path.join(output_dir, 'model_tokenizer.pkl')
        with open(temp_path, 'wb') as f:
            dill.dump(self.model_tokenizer, f)

        # save configuration
        self.configuration.save_pretrained(output_dir)

    def _load_tokenizer(self, data_tokenizer_file, model_tokenizer_file):
        with open(data_tokenizer_file, 'rb') as f:
            self.data_tokenizer = dill.load(f)
        self.data_tokenizer._in_target_context_manager = False # fix bugs when upgrade transformers to 4.23

        with open(model_tokenizer_file, 'rb') as f:
            self.model_tokenizer = dill.load(f)

    def _load_config(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        return config

    def _get_test_dataloader(self, dataset):
        def _seq_patient_to_promptehr(samples):
            post_samples = []
            for sample in samples:
                post_sample = {}
                visit = sample['v']
                post_sample.update(visit)

                if ('x' in sample) and (self.config['n_num_feature'] is not None):
                    if not isinstance(sample['x'], list): 
                        sample['x'] = sample['x'].tolist()
                    post_sample['x_num'] = torch.tensor(sample['x'][:self.config['n_num_feature']])
                    post_sample['x_cat'] = torch.tensor(sample['x'][self.config['n_num_feature']:], dtype=int)

                if 'y' in sample:
                    post_sample['y'] = sample['y']

                post_samples.append(post_sample)
            return post_samples

        dataloader = DataLoader(dataset,
                batch_size=1, # one patient once
                drop_last=False,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                collate_fn=_seq_patient_to_promptehr,
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
            filepath = os.path.join(output_dir, str(epoch_id) + '.' + filename)
        
        # save statedict
        state_dict = self.state_dict()
        torch.save(state_dict, filepath)

    def _fit(self, train_data, val_data):
        mimic_train_collator = MimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            max_train_batch_size=self.config['batch_size'], mode='train')

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
                self.data_tokenizer.add_token_to_code_vocab(unq_codes, k)
        
        self.model_tokenizer = ModelTokenizer(self.data_tokenizer)
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=self.config['n_num_feature'], cat_cardinalities=self.config['cat_cardinalities'])
        self.data_tokenizer.update_special_token_config(code_types=self.config['code_type'])

        # self.data_tokenizer.decode([50508, 51324,51461, 50597, 50918,]) 


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
            return n, math.ceil(n / n_test_sample)

    def _get_num_visit(self, data, idx):
        visits = data['v']
        num_visit_list = []
        for k,v in visits.items():
            num_visit_list.append(len(v[idx]))
        
        num_visit_uq = list(set(num_visit_list))
        assert len(num_visit_uq) == 1, f'Find mismatch in the number of visit events {num_visit_list}, please check the input data {visits}.'
        return num_visit_uq[0]

    def _predict_on_dataloader(self, dataloader, n, n_per_sample, verbose=None):
        total_number = 0
        data_iterator = iter(dataloader)

        if verbose:
            pbar = tqdm(total=n)

        new_record_list = []
        while total_number < n:
            try:
                data = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                data = next(data_iterator)            
            data = data[0] # batch size is 1 when doing generation

            # to device
            device = 'cpu' if self.device == 'cpu' else 'cuda:0'
            if 'x_num' in data: data['x_num'] = data['x_num'].to(device)
            if 'x_cat' in data: data['x_cat'] = data['x_cat'].to(device)                
            
            inputs = self._prepare_input_for_generation(data) 

            # start generation
            for _ in range(n_per_sample):
                new_record = self._generation_loop(data, inputs)
                if 'x_cat' in data:
                    new_record.update({
                        'x_cat':data['x_cat'].cpu().numpy().tolist(),
                    })
                    
                if 'x_num' in data:
                    new_record.update({
                        'x_num':data['x_num'].cpu().numpy().tolist(),
                    })

                # add more features to new_record
                for k,v in data.items():
                    if k not in new_record:
                        new_record[k] = v
                new_record_list.append(new_record)
            
            total_number += n_per_sample
            if verbose:
                pbar.update(n_per_sample)
                    
        if verbose:
            pbar.close()
        return new_record_list

    def _prepare_input_for_generation(self, data):        
        def _process_span(span, code):
            return [code+'_'+str(s) for s in span]
        
        def _to_device(x, device):
            for k,v in x.items():
                x[k] = v.to(device)
            return x

        tokenizer = self.data_tokenizer
        code_type = [k for k in data.keys() if k in self.config['code_type']]
        num_visit = len(data[code_type[0]])
        
        # init codes
        init_code = random.sample(data[code_type[0]][0], 1)
        init_code_str = _process_span(init_code, code_type[0])
        init_codes = tokenizer(init_code_str, return_tensors='pt', add_special_tokens=False)
        bos = torch.tensor([tokenizer.bos_token_id])
        code_prompt_idx = tokenizer.encode(tokenizer.special_token_dict[code_type[0]], add_special_tokens=False, return_tensors='pt')
        init_input_ids = torch.cat([bos[:,None],code_prompt_idx[:,0,None],init_codes['input_ids']], dim=-1)
        init_input_ids = _to_device({'input_ids':init_input_ids}, self.model.device)['input_ids']
        input_ids = init_input_ids.clone()
        return {'input_ids':input_ids, 'init_input_ids':init_input_ids, 'num_visit':num_visit, 'init_code':init_code}

    def _generation_loop(self, data, inputs):
        new_record = defaultdict(list)
        tokenizer = self.data_tokenizer
        special_token_dict = self.data_tokenizer.special_token_dict
        sample_gen_kwargs = self.sample_config.copy()

        input_ids_list = []
        num_visit_code_list = []
        first_code_flag = True

        input_ids = inputs['input_ids']
        for visit in range(inputs['num_visit']):
            this_visit_ids_list = []
            for code in self.config['code_type']:
                target_list = data[code][visit]
                sample_gen_kwargs['code_type'] = code
                num_code = len(target_list)
                if num_code > 20:
                    num_code = min(num_code, 20)
                    target_list = np.random.choice(target_list, num_code, replace=False).tolist()

                # random select part of codes from target list
                target_ar = np.array(target_list)
                sub_code = target_ar[np.random.binomial(1, 0.5, num_code).astype(bool)]
                code_prompt_idx = [special_token_dict[code][0]] + sub_code.tolist() + [special_token_dict[code][1]]
                code_prompt_idx = tokenizer.encode(code_prompt_idx, add_special_tokens=False, return_tensors='pt')
                code_prompt_idx = code_prompt_idx.to(self.model.device)

                if num_code == 0:
                    if first_code_flag:
                        new_next_tokens = code_prompt_idx[:,-1,None]
                        first_code_flag = False
                    else:
                        new_next_tokens = code_prompt_idx

                    this_visit_ids_list.append(new_next_tokens)
                    input_ids = torch.cat([input_ids, new_next_tokens], dim=-1)
                    new_record[code].append([])
                
                else:
                    sample_gen_kwargs['max_length'] = num_code+2

                    # do conditional generation
                    if 'x_cat' in data:
                        sample_gen_kwargs['x_cat'] = data['x_cat']
                    if 'x_num' in data:
                        sample_gen_kwargs['x_num'] = data['x_num']

                    new_next_tokens = self.model.generate(input_ids, **sample_gen_kwargs)

                    # randomly pick / rm sub code overlap
                    new_next_tokens = new_next_tokens[:,1:-1]
                    new_next_tokens = np.setdiff1d(new_next_tokens[0].cpu(), code_prompt_idx[0].cpu())
                    if num_code-len(sub_code) > len(new_next_tokens):
                        new_sub_idxs = np.unique(np.random.choice(np.arange(len(new_next_tokens)), num_code-len(sub_code), replace=True))
                    else:
                        new_sub_idxs = np.unique(np.random.choice(np.arange(len(new_next_tokens)), num_code-len(sub_code), replace=False))
                    new_next_tokens = torch.tensor(new_next_tokens[None, new_sub_idxs]).to(code_prompt_idx.device)

                    # append to the synthetic record dict
                    code_str_list = tokenizer.batch_decode(new_next_tokens)[0]

                    # remove special tokens ahead of original code event
                    # e.g., `diag_384` -> `384`
                    code_str_list = code_str_list.replace(code+'_','')
                    code_str_list = code_str_list.split()
                    code_str_list = [int(c) for c in code_str_list+sub_code.tolist()]
                    new_record[code].append(list(set(code_str_list)))

                    if first_code_flag:
                        new_next_tokens = torch.cat([new_next_tokens, code_prompt_idx[:,1:]], dim=-1)
                        first_code_flag = False
                    else:
                        # cover by modality prompt
                        new_next_tokens = torch.cat([code_prompt_idx[:,:-1], new_next_tokens, code_prompt_idx[:,-1,None]], dim=-1)

                    if visit > 1:
                        # check input length
                        cur_len = input_ids.shape[1] + new_next_tokens.shape[1]
                        while cur_len >= tokenizer.model_max_length:
                            print(f'{cur_len} reach model max length {tokenizer.model_max_length}, do cut.')
                            input_ids_list = input_ids_list[1:]
                            num_visit_code_list = num_visit_code_list[1:]
                            input_ids = torch.cat(input_ids_list,dim=-1)
                            cur_len = input_ids.shape[1] + new_next_tokens.shape[1]

                    # concat
                    this_visit_ids_list.append(new_next_tokens)
                    input_ids = torch.cat([input_ids, new_next_tokens], dim=-1)

            # after one visit, add eos token id
            eos = torch.tensor([tokenizer.eos_token_id], device=self.model.device)
            input_ids = torch.cat([input_ids, eos[:,None]], dim=-1)
            this_visit_ids = torch.cat(this_visit_ids_list, dim=-1)
            this_visit_ids = torch.cat([this_visit_ids, eos[:,None]], dim=-1)
            if visit == 0: this_visit_ids = torch.cat([inputs['init_input_ids'], this_visit_ids], dim=-1)
            num_visit_code_list.append(this_visit_ids.shape[-1])
            input_ids_list.append(this_visit_ids)

        # add init code
        new_record[self.config['code_type'][0]][0] += inputs['init_code']
        return new_record


def download_pretrained(url, output_dir):
    import wget
    import zipfile
    filename = wget.download(url=url, out=output_dir)
    zipf = zipfile.ZipFile(filename, 'r')
    zipf.extractall(output_dir)
    zipf.close()

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
    model_config_name = [config for config in ckpt_list if 'promptehr_config.json' in config]
    if len(model_config_name) == 1:
        return model_config_name[0]

    # if no model_config.json found, retrieve the only .json file.
    assert len(ckpt_list) <= 1, f'Find more than one config .json under the dir {input_dir}.'
    return ckpt_list[0]


def check_tokenizer_file(input_dir):
    return os.path.join(input_dir,'data_tokenizer.pkl'), os.path.join(input_dir,'model_tokenizer.pkl')

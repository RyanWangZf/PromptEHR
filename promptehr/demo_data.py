import os
import json
import pdb
import wget
import dill

from .dataset import MimicDataset
from .import constants

# def load_demo_data(input_dir='./demo_data'):
#     '''
#     Load demo data.
#     '''
#     return_dict = {}

#     with open(os.path.join(input_dir, 'token_dict.json'), 'r', encoding='utf-8') as f:
#         token_dict = json.loads(f.read())
    
#     return_dict['token_dict'] = token_dict

#     with open(os.path.join(input_dir, 'cat_cardinalities.txt'), 'r', encoding='utf-8') as f:
#         cats = [l.strip() for l in f.readlines()]
#     return_dict['cat_cardinalities'] = [int(c) for c in cats]
#     return_dict['n_num_feature'] = 1

#     mimic_dataset = MimicDataset(input_dir, mode='train')
#     mimic_val_dataset = MimicDataset(input_dir, mode='val')

#     return_dict['dataset'] = mimic_dataset
#     return_dict['val_dataset'] = mimic_val_dataset

#     return return_dict

def load_synthetic_data(input_dir='./demo_data/synthetic_ehr', n_sample=None):
    '''
    Load the generated synthetic EHRs by PromptEHR.
    '''
    if input_dir is None or not os.path.exists(input_dir):
        if input_dir is None:
            input_dir = './demo_data/synthetic_ehr'
        os.makedirs(input_dir)
        url = constants.SYNTHETIC_DATA_URL
        filename = wget.download(url, out=input_dir)
        print(f'Download synthetic EHRs to {input_dir}.')
    
    with open(os.path.join(input_dir,'data.pkl'), 'rb') as f:
        x = dill.load(f)

    if n_sample is not None:
        # cut to get smaller demo data
        x['visit'] = x['visit'][:n_sample]
        x['y'] = x['y'][:n_sample]
        x['feature'] = x['feature'][:n_sample]

    return x


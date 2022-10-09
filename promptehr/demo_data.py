import os
import json
import pdb

from .dataset import MimicDataset

def load_demo_data(input_dir='./demo_data'):
    '''
    Load demo data.
    '''
    return_dict = {}

    with open(os.path.join(input_dir, 'token_dict.json'), 'r', encoding='utf-8') as f:
        token_dict = json.loads(f.read())
    
    return_dict['token_dict'] = token_dict

    with open(os.path.join(input_dir, 'cat_cardinalities.txt'), 'r', encoding='utf-8') as f:
        cats = [l.strip() for l in f.readlines()]
    return_dict['cat_cardinalities'] = [int(c) for c in cats]
    return_dict['n_num_feature'] = 1

    mimic_dataset = MimicDataset(input_dir, mode='train')
    mimic_val_dataset = MimicDataset(input_dir, mode='val')

    return_dict['dataset'] = mimic_dataset
    return_dict['val_dataset'] = mimic_val_dataset

    return return_dict



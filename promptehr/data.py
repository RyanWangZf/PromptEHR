import warnings
import pdb
import json
import pickle
import dill
from collections import defaultdict
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
    
    def __len__(self):
        return len(self.idx2word.keys())

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

class SequencePatientBase(Dataset):
    '''
    Load sequential patient inputs for longitudinal patient records generation.
    Parameters
    ----------
    data: dict
        A dict contains patient data in sequence and/or in tabular.
        Given dict:
            {
                'x': np.ndarray or pd.DataFrame
                    Static patient features in tabular form, typically those baseline information.
                'v': list or np.ndarray
                    Patient visit sequence in dense format or in tensor format (depends on the model input requirement.)
                    If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];
                    If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), 
                        with shape [n_patient, max_num_visit, max_num_event].
                
                'y': np.ndarray or pd.Series
                    Target label for each patient if making risk detection, with shape [n_patient, n_class];
                    Target label for each visit if making some visit-level prediction with shape [n_patient, NA, n_class].
            }
    
    metadata: dict (optional)
        A dict contains configuration of input patient data.
        metadata:
            {
                'voc': dict[Voc]
                    Vocabulary contains the event index to the exact event name, has three keys in general:
                    'diag', 'med', 'prod', corresponding to diagnosis, medication, and procedure.
                    ``Voc`` object should have two functions: `idx2word` and `word2idx`.
                
                'visit': dict[str]
                    a dict contains the format of input data for processing input visit sequences.
                    `visit`: {
                        'mode': 'tensor' or 'dense',
                        'order': list[str] (required when `mode='tensor'`)
                    },
                'label': dict[str]
                    a dict contains the format of input data for processing input labels.
                    `label`: {
                        'mode': 'tensor' or 'dense',
                    }
                
                'max_visit': int
                    the maximum number of visits considered when building tensor inputs, ignored
                    when visit mode is dense.
            }
    '''
    visit = None
    feature = None
    label = None
    max_visit = None
    visit_voc_size = None
    visit_order = None

    metadata = {
        'voc': {},
        
        'visit':{
            'mode': 'dense',
            'order': ['diag', 'prod', 'med'],
            },

        'label':{
            'mode': 'tensor',
            },

        'max_visit': 20,
    }

    def __init__(self, data, metadata=None) -> None:
        # parse metadata
        self._parse_metadata(metadata=metadata)

        # get input data
        self._parse_inputdata(inputs=data)

    def __getitem__(self, index):
        return_data = {}
        if self.visit is not None:
            visits = self.visit[index]
            if self.metadata['visit']['mode'] == 'tensor':
                visit_ts = self._dense_visit_to_tensor(visits) # return a dict with keys corresponding to order
                return_data['v'] = visit_ts
            else:
                visit_dict = self._parse_dense_visit_with_order(visits) # return a dict with keys corresponding to otder
                return_data['v'] = visit_dict
        
        if self.feature is not None:
            return_data['x'] = self.feature[index]
            
        if self.label is not None:
            return_data['y'] = self.label[index]
        
        return return_data

    def __len__(self):
        return len(self.visit)

    def _get_voc_size(self):
        order = self.metadata['visit']['order']
        vocs = self.metadata['voc']
        voc_size = []
        for order_ in order:
            voc_size.append(
                len(vocs[order_])
            )
        self.visit_voc_size = voc_size

    def _read_pickle(self, file_loc):
        return dill.load(open(file_loc, 'rb'))
    
    def _parse_metadata(self, metadata):
        if metadata is not None: 
            for k,v in metadata.items():
                if isinstance(v, dict):
                    self.metadata[k].update(v)
                else:
                    self.metadata[k] = v
        metadata = self.metadata

        if 'voc' in metadata:
            voc = metadata['voc']

            if 'diag' in voc: self.diag_voc = voc['diag']
            if 'prod' in voc: self.prod_voc = voc['prod']
            if 'med' in voc: self.med_voc = voc['med']
        
        if metadata['visit']['mode'] == 'tensor':
            self._get_voc_size()
        
        if 'order' in metadata['visit']:
            self.visit_order = metadata['visit']['order']

        if 'max_visit' in metadata:
            self.max_visit = metadata['max_visit']

    def _parse_inputdata(self, inputs):
        if 'x' in inputs: self.feature = inputs['x']
        if 'v' in inputs: self.visit = inputs['v']
        if 'y' in inputs: self.label = inputs['y']

    def _dense_visit_to_tensor(self, visits):
        res = {}
        for i,o in enumerate(self.visit_order):
            res[o] = np.zeros((self.max_visit, self.visit_voc_size[i]), dtype=int)

        for i, visit in enumerate(visits):
            # clip if the max visit is larger than self.max_visit
            if i >= self.max_visit: break
            
            for j, o in enumerate(self.visit_order):
                res[o][i, visit[j]] = 1
        return res
    
    def _parse_dense_visit_with_order(self, visits):
        return_data = defaultdict(list)
        order_list = self.metadata['visit']['order']
        for visit in visits:
            for i, o in enumerate(order_list):
                return_data[o].append(visit[i])
        return return_data


class SequencePatient(SequencePatientBase):
    '''
    Load sequential patient inputs for longitudinal patient records generation.
    Parameters
    ----------
    data: dict
        A dict contains patient data in sequence and/or in tabular.
        Given dict:
            {
                'x': np.ndarray or pd.DataFrame
                    Static patient features in tabular form, typically those baseline information.
                'v': list or np.ndarray
                    Patient visit sequence in dense format or in tensor format (depends on the model input requirement.)
                    If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];
                    If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), 
                        with shape [n_patient, max_num_visit, max_num_event].
                
                'y': np.ndarray or pd.Series
                    Target label for each patient if making risk detection, with shape [n_patient, n_class];
                    Target label for each visit if making some visit-level prediction with shape [n_patient, NA, n_class].
            }
    
    metadata: dict (optional)
        A dict contains configuration of input patient data.
        metadata:
            {
                'voc': dict[Voc]
                    Vocabulary contains the event index to the exact event name, has three keys in general:
                    'diag', 'med', 'prod', corresponding to diagnosis, medication, and procedure.
                    ``Voc`` object should have two functions: `idx2word` and `word2idx`.
                
                'visit': dict[str]
                    a dict contains the format of input data for processing input visit sequences.
                    `visit`: {
                        'mode': 'tensor' or 'dense',
                        'order': list[str] (required when `mode='tensor'`)
                    },
                'label': dict[str]
                    a dict contains the format of input data for processing input labels.
                    `label`: {
                        'mode': 'tensor' or 'dense',
                    }
                
                'max_visit': int
                    the maximum number of visits considered when building tensor inputs, ignored
                    when visit mode is dense.
            }
    '''
    visit = None
    feature = None
    label = None
    max_visit = None
    visit_voc_size = None
    visit_order = None

    metadata = {
        'voc': {},
        
        'visit':{
            'mode': 'dense',
            'order': ['diag', 'prod', 'med'],
            },

        'label':{
            'mode': 'tensor',
            },

        'max_visit': 20,
    }

    def __init__(self, data, metadata=None) -> None:
        super().__init__(data=data, metadata=metadata)
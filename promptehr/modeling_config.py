from collections import defaultdict
import warnings
import pdb, os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass
import torch
import numpy as np
from transformers import BartTokenizer
from transformers import BartConfig
from transformers.file_utils import ModelOutput
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel

from . import constants

def EHRBartConfig(data_tokenizer, model_tokenizer, **kwargs):
    '''Build the config used for building the promptBart model.
    '''
    bart_config = BartConfig.from_pretrained('facebook/bart-base')
    kwargs.update(model_tokenizer.get_num_tokens)
    kwargs['data_tokenizer_num_vocab'] = len(data_tokenizer)
    if 'd_prompt_hidden' not in kwargs:
        kwargs['d_prompt_hidden'] = 128
    if 'n_num_feature' not in kwargs:
        kwargs['n_num_feature'] = None
    if 'cat_cardinalities' not in kwargs:
        kwargs['cat_cardinalities'] = None
    bart_config.__dict__.update(kwargs)

    # specify bos, eos token id
    bart_config.__dict__['decoder_start_token_id'] = 0
    bart_config.__dict__['bos_token_id'] = 0
    bart_config.__dict__['eos_token_id'] = 1
    bart_config.__dict__['forced_eos_token_id'] = 1
    return bart_config

class DataTokenizer(BartTokenizer):
    r'''construct tokenizer to process the input raw records.
    '''
    new_token_type_list = constants.CODE_TYPES
    special_token_dict = constants.SPECIAL_TOKEN_DICT
    code_vocab = defaultdict(list)

    def add_token_to_code_vocab(self, tokens, code):
        self.add_tokens(tokens)

        if code not in self.code_vocab:
            self.code_vocab[code] = np.array(tokens)
        else:
            origin_tokens = self.code_vocab[code]
            new_tokens = np.array(tokens)
            self.code_vocab[code] = np.unique(np.concatenate([origin_tokens, new_tokens]))

    def update_special_token_config(self, code_types):
        self.new_token_type_list = code_types
        self.special_token_dict = {}
        special_token_list = []
        for code_type in code_types:
            l = [f'<{code_type}>', f'</{code_type}>']
            self.special_token_dict[code_type] = l
            special_token_list.extend(l)
        self.add_tokens(special_token_list)

    def extend_vocab(self, token_dict):
        '''
        Parameters:
        ----------
        token_dict: dict
            key: code type, value: a list of tokens.
        '''
        for key in token_dict.keys():
            self.code_vocab[key] = np.array(token_dict[key])
            self.add_tokens(token_dict[key])

    def extend_vocab_from_dir(self, data_dir):
        # add new tokens from the data dir
        for key in self.new_token_type_list:
            filename = os.path.join(data_dir,'{}_token_list.txt'.format(key))
            with open(filename, 'r', encoding='utf-8') as f:
                token_list = [line.strip() for line in f.readlines()]
            self.code_vocab[key] = np.array(token_list)
            self.add_tokens(token_list)

        # add special tokens indicating different modality
        for key, value in self.special_token_dict.items():
            self.add_tokens(value, special_tokens=True)

class ModelTokenizer:
    r'''construct an EHR tokenizer that converts tokenized indices to code-specific token indices.
    '''
    def __init__(self, tokenizer: DataTokenizer):
        # map_token = lambda x: str(tokenizer(x).input_ids[1])
        org_vocab = tokenizer.get_vocab()
        tokenizer_dict = {}
        num_token_dict = {}
        for key, value in tokenizer.code_vocab.items():
            vocab = defaultdict(int)
            vocab[constants.UNKNOWN_TOKEN] = 0
            for i,token in enumerate(tokenizer.special_token_dict[key]):
                vocab[str(org_vocab[token])] = i+1
            offset = len(vocab)

            for i, token in enumerate(value): # str token = 'diag_xxx'
                # fix: if token has more than one '_', e.g., 'diag_t_a_b_100', will only take the last '100' as the index. 
                # _, index = token.split('_')
                indexes = token.split('_')
                try:
                    index = int(indexes[-1])
                except:
                    raise ValueError(f"Token {token} is not a valid token, it should be splited by '_' and the last part should be a number, e.g., 'diag_100'. ")
                vocab[str(org_vocab[token])] = index + offset
            
            # new tokenizer
            specific_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=constants.UNKNOWN_TOKEN))
            specific_tokenizer.pre_tokenizer = Whitespace()

            # num_token_dict is decided by the max index instead of number of tokens
            num_token_dict[key] = (max(vocab.values())+1) - offset
            tokenizer_dict[key] = specific_tokenizer

        # each code type has its own tokenizer corresponding to specific LM heads
        self.tokenizer_dict = tokenizer_dict
        self.num_token_dict = num_token_dict
        self.label_offset = offset

    def encode(self, input_ids, code_type):
        if len(input_ids.shape) > 1: # a batch
            ids = self.encode_batch(input_ids, code_type)
        else:
            ids = self.tokenizer_dict[code_type].encode(input_ids.cpu().numpy().astype(str), is_pretokenized=True).ids
            ids = torch.tensor(ids, device=input_ids.device)
        return ids

    def encode_batch(self, input_ids, code_type):
        ids_list = self.tokenizer_dict[code_type].encode_batch(input_ids.cpu().numpy().astype(str).tolist(), is_pretokenized=True)

        ids = torch.tensor([x.ids for x in ids_list], device=input_ids.device)
        return ids

    @property
    def get_num_tokens(self):
        return self.num_token_dict

@dataclass
class EHRBartOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        perplexity:
            perplexity calculated when the label mask is given.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    perplexity: Optional[torch.FloatTensor] = None

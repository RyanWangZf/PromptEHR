import pdb
import random
import logging
import warnings
from typing import Optional, List

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, BartEncoder, BartDecoder
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
import torch
from torch import nn
from torch import Tensor

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

class NumericalConditionalPrompt(nn.Module):
    '''Embedding for conditional prompts based on numerical input patient features,
    take reparametrization trick.

    Parameters
    ----------
    n_feature: number of input features.
    d_model: dimension of output embeddings.
    d_hidden: dimension of intermediate embeddings for reparametrization.
    '''
    def __init__(self, n_feature, d_model, d_hidden) -> None:
        super().__init__()
        self.weight = nn.init.xavier_uniform_(nn.Parameter(Tensor(n_feature, d_hidden)))
        self.bias = nn.init.xavier_uniform_(nn.Parameter(Tensor(n_feature, d_hidden)))
        self.proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        x = self.weight[None] * x[..., None]
        x = x + self.bias[None]
        x = self.proj(x)
        return x

class CategoricalConditionalPrompt(nn.Module):
    '''Embedding for conditional prompts based on categorical input patient features,
    take reparametrization trick.

    Parameters
    ----------
    cardinalities: the number of distinct values for each feature, e.g., [2, 3, 5] indicates the first cat has 2 possible categories and so on.
    d_model: the output embedding dimension.
    d_hidden: the intermediate layer dimension for reparameterization.
    '''
    def __init__(self,
        cardinalities,
        d_model,
        d_hidden
        ) -> None:
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_hidden)
        self.bias = nn.init.xavier_uniform_(nn.Parameter(Tensor(len(cardinalities),d_hidden)))
        self.proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        x = self.embeddings(x + self.category_offsets[None])
        x = x + self.bias[None]
        x = self.proj(x)
        return x

class ConditionalPrompt(nn.Module):
    '''Provide conditional prompt embedding for both categorical and numerical features.

    Parameters
    ----------
    n_num_feature: number of input numerical features.
    cat_cardinalities: a list of unique numbers of each feature.
    d_model: the output dimension.
    d_hidden: the intermediate layer dimension for reparametrization.
    '''
    def __init__(self,
        n_num_feature=None,
        cat_cardinalities=None,
        d_model=None,
        d_hidden=None,
        ) -> None:
        super().__init__()
        if n_num_feature is not None: 
            assert isinstance(n_num_feature, int), 'the passed `n_num_feature` to `promptehr` must be an integer, {} with type {} found.'.format(n_num_feature, type(n_num_feature))
            assert n_num_feature >= 0, 'n_num_feature must be non-negative'
        assert (n_num_feature or cat_cardinalities), 'at least one of n_num_feature or cat_cardinalities must be positive/non-empty'
        self.num_tokenizer = (
            NumericalConditionalPrompt(
                n_feature=n_num_feature,
                d_model=d_model,
                d_hidden=d_hidden,
            )
            if n_num_feature
            else None
        )
        self.cat_tokenizer = (
            CategoricalConditionalPrompt(
                cat_cardinalities,
                d_model=d_model,
                d_hidden=d_hidden,
            )
            if cat_cardinalities
            else None
        )

    def forward(self, x_num=None, x_cat=None):
        '''Perform the forward pass to encode features into prompt context vectors.

        Parameters
        ----------
        x_num: continuous features. Must be presented if :code:`n_num_feature > 0` was passed.
        x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed.
        '''
        assert (
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

class PromptBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        embed_dim = config.d_model
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor]=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        '''Make encoding.
        Parameters
        ----------
        inputs_prompt_embeds: Embeded conditional prompt embeddings.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if inputs_prompt_embeds is not None:
            # concatenate prompt embeddings in front of the input embeds
            # modify input_shape and attention_mask at the same time
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            input_shape = inputs_embeds.size()[:-1]
            if attention_mask is not None:
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(attention_mask.device)
                attention_mask = torch.cat([add_att_mask, attention_mask], dim=1)

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class PromptBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        ):
        '''Make forward pass by the decoder.

        Parameters
        ----------
        inputs_prompt_embeds: the embeddings of conditional prompts for the decoder.

        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if inputs_prompt_embeds is not None:
            # concatenate prompt embeddings in front of the input embeds
            # modify input_shape and attention_mask at the same time
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            input_shape = inputs_embeds.size()[:-1]
            if attention_mask is not None:
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(attention_mask.device)
                attention_mask = torch.cat([add_att_mask, attention_mask], dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if inputs_prompt_embeds is not None:
                # adjust for input prompt embeddings
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([add_att_mask, encoder_attention_mask], dim=1)

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "[warning] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else: # testing/generating                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class PromptBartModel(BartModel):
    '''a subclass of BartModel by using additional prompts for controllable EHR generation.
    '''
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PromptBartEncoder(config, self.shared)
        self.decoder = PromptBartDecoder(config, self.shared)

        # build encoder & decoder prompts
        n_num_feature = config.n_num_feature
        cat_cardinalities = config.cat_cardinalities
        if n_num_feature is not None or cat_cardinalities is not None:
            self.encoder_conditional_prompt = ConditionalPrompt(n_num_feature=n_num_feature,
                cat_cardinalities=cat_cardinalities,
                d_model=config.d_model,
                d_hidden=config.d_prompt_hidden)
            self.decoder_conditional_prompt = ConditionalPrompt(n_num_feature=n_num_feature,
                cat_cardinalities=cat_cardinalities,
                d_model=config.d_model,
                d_hidden=config.d_prompt_hidden)
        else:
            # fix when no baseline feature is provided.
            warnings.warn('No numerical or categorical baseline features are provided, `ConditionalPrompt` is not used in the model.')
            self.encoder_conditional_prompt = None
            self.decoder_conditional_prompt = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        '''Make the forward pass to encode inputs with Bart model.

        Parameters
        ----------
        x_num: the input numerical features, shape (bs, num_feat)
        x_cat: the input categorical features, shape (bs, num_cat)
        '''
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if x_num is not None or x_cat is not None:
                if self.encoder_conditional_prompt is None:
                    warnings.warn('Detect input baseline features in the data,` \
                        but `ConditionalPrompt was not built because no numerical or categorical baseline features are provided when model was initialized. \
                        Consider setting `config.n_num_feature` or `config.cat_cardinalities` when initializing the model.')
                    prompt_embeds = None
                else:
                    prompt_embeds = self.encoder_conditional_prompt(x_num=x_num, x_cat=x_cat)
            else:
                prompt_embeds = None
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                inputs_prompt_embeds=prompt_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if x_num is not None or x_cat is not None:
            if self.decoder_conditional_prompt is None:
                warnings.warn('{} {} {}'.format('Detect input baseline features in the data, but `ConditionalPrompt`',
                    'was not built because no numerical or categorical baseline features',
                    'Consider setting `config.n_num_feature` or `config.cat_cardinalities` when initializing the model.')
                )
                decoder_prompt_embeds = None 
            else:
                decoder_prompt_embeds = self.decoder_conditional_prompt(x_num=x_num, x_cat=x_cat)
        else:
            decoder_prompt_embeds = None
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            inputs_prompt_embeds=decoder_prompt_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
from cmath import isnan
import pdb
import warnings

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
from torch import nn

from .generator import EHRGenerationMixin
from .modeling_config import EHRBartOutput
from .modeling_config import DataTokenizer, ModelTokenizer
from .modeling_config import EHRBartConfig
from .modeling_promptbart import PromptBartModel
from . import constants

class BartForEHRSimulation(BartPretrainedModel, EHRGenerationMixin):
    def __init__(self, config: EHRBartConfig, model_tokenizer: ModelTokenizer=None, data_tokenizer: DataTokenizer=None):
        super().__init__(config)

        config.is_decoder = False
        config.is_encoder_decoder = True # use both the inputs for encoders and decoders

        self.model = PromptBartModel(config)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.lm_head = {}
        for key in model_tokenizer.tokenizer_dict.keys():
            self.lm_head[key] =  nn.Linear(
                config.d_model, config.__dict__[key], bias=False,
            )
        self.lm_head = nn.ModuleDict(self.lm_head)

        self.init_weights()

        # set embedding vocab new size
        self.resize_token_embeddings(config.data_tokenizer_num_vocab)

        # for specific modal generation
        self.model_tokenizer = model_tokenizer
        self.data_tokenizer = data_tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        x_num = None,
        x_cat = None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        label_mask=None,
        code_type=None,
    ):
        r"""This forward is specified for *ALL* possible words in the vocabulary. It's same as the BartForConditionalGeneration.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        code_types = list(self.model_tokenizer.tokenizer_dict.keys())
        assert code_type in code_types

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id,
                )
        
        # get the outputs from both encoder and decoder of BartModel
        # if only input_ids are given and no decoder_input_ids is given
        # the model will shift input_ids to get the decoder input ids during forward
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask, # 0 for"masked", 1 for "Not masked"
            decoder_input_ids=decoder_input_ids,
            x_num=x_num,
            x_cat=x_cat,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # for diagnosis/procedure/lab tests/medications
        logits = self.lm_head[code_type](outputs[0])

        loss = None
        perplexity = None
        if labels is not None:
            encoded_labels = self.model_tokenizer.encode(labels, code_type)
            encoded_labels = encoded_labels - self.model_tokenizer.label_offset
            encoded_labels[encoded_labels < 0] = -100 # ignore special tokens when computing losses

            if (x_num is not None or x_cat is not None) \
                and (self.model.encoder_conditional_prompt is not None) \
                and (self.model.decoder_conditional_prompt is not None):
                # adjust label mask if context conditional prompt is used

                n_num = x_num.shape[1] if x_num is not None else 0
                n_cat = x_cat.shape[1] if x_cat is not None else 0
                num_label_offset = n_num + n_cat
                label_offset = torch.ones(encoded_labels.shape[0], num_label_offset) * -100
                label_offset = label_offset.long().to(labels.device)
                encoded_labels = torch.cat([label_offset, encoded_labels], dim=1)
                if label_mask is not None:
                    label_mask_offset = torch.zeros(encoded_labels.shape[0], num_label_offset)
                    label_mask_offset = label_mask_offset.to(label_mask.device)
                    label_mask = torch.cat([label_mask_offset, label_mask], dim=1)

            # compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.lm_head[code_type].out_features), encoded_labels.view(-1))
            
            if label_mask is not None: # do evaluation, compute perplexity
                if encoded_labels[encoded_labels >= 0].shape[0] == 0:
                    perplexity = None
                else:
                    target = encoded_labels[label_mask.bool()]
                    mask_logits = logits[label_mask.bool()]

                    # debug: move to CPU see errors
                    # prob = torch.gather(mask_logits.softmax(1).cpu(), 1, target.unsqueeze(-1).cpu())

                    prob = torch.gather(mask_logits.softmax(1), 1, target.unsqueeze(-1))
                    nll = -torch.log(prob+constants.eps)
                    perplexity = nll.exp()
                    if torch.isnan(perplexity).any():
                        warnings.warn('Find NaN perplexity during the forward of PromptEHR model!')
                    perplexity = perplexity.median()
                
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # debug
        # if logits.isnan().sum() > 0:
        #     pdb.set_trace()

        return EHRBartOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            perplexity=perplexity,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        return_res = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if 'x_cat' in kwargs: return_res['x_cat'] = kwargs['x_cat']
        if 'x_num' in kwargs: return_res['x_num'] = kwargs['x_num']

        return return_res

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_prompt_encoder(self):
        return self.model.encoder_conditional_prompt
    
    def get_prompt_decoder(self):
        return self.model.decoder_conditional_prompt

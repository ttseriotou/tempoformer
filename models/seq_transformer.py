import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import BertPreTrainedModel, BertConfig, RobertaPreTrainedModel, RobertaConfig 
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaPooler
from transformers.utils.doc import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
) 
from transformers.models.bert.modeling_bert import (BERT_INPUTS_DOCSTRING,
                                                    _CHECKPOINT_FOR_DOC as BERT_CHECKPOINT_FOR_DOC,
                                                    _CONFIG_FOR_DOC as BERT_CONFIG_FOR_DOC
)
from transformers.models.roberta.modeling_roberta import (ROBERTA_INPUTS_DOCSTRING,
                                                    _CHECKPOINT_FOR_DOC as ROBERTA_CHECKPOINT_FOR_DOC,
                                                    _CONFIG_FOR_DOC as ROBERTA_CONFIG_FOR_DOC
)
from models.seq_encoder import SeqBertEncoder, SeqRobertaEncoder
from models.ffn import FeedforwardNeuralNetModel

class SeqBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config,
                position_seqlayer: list[int] | tuple[int] | int,
                window: int,
                pooling: str,
                pos_word_embedding: str | None,
                connection: str| None,
                add_pooling_layer=True,
                ):
        super().__init__(config)
        self.config = config

        if isinstance(position_seqlayer, int):
            position_seqlayer = [position_seqlayer]

        self.embeddings = BertEmbeddings(config)
        self.encoder = SeqBertEncoder(config, position_seqlayer, 
                                    window, pooling, 
                                    pos_word_embedding, connection)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.pooler_window = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=BERT_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=BERT_CONFIG_FOR_DOC,
    )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, _, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #flatten timeline dimension into batch dimension
        attn_mask_shape = input_ids[:,0,:].size()
        input_ids = torch.flatten(input_ids,0,1)
        token_type_ids = torch.flatten(token_type_ids,0,1)
        attention_mask = torch.flatten(attention_mask,0,1)
        zero_mask = torch.flatten(zero_mask,0,1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, attn_mask_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None: #ttseriotou
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            zero_mask=zero_mask,
            time=time,
            stream_dims = input_shape,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0][:,-1,:,:]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        pooled_window_output = encoder_outputs[1][-1][:,-1,:]
        pooled_output = torch.cat((pooled_output, pooled_window_output), 1)
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] #HERE
            #return _, _ , encoder_outputs[0][-1] 
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class SeqRobertaModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config,
                position_seqlayer: list[int] | tuple[int] | int,
                window: int,
                pooling: str,
                pos_word_embedding: str | None,
                connection: str| None,
                add_pooling_layer=True,
                ):
        super().__init__(config)
        self.config = config

        if isinstance(position_seqlayer, int):
            position_seqlayer = [position_seqlayer]

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = SeqRobertaEncoder(config, position_seqlayer, 
                                    window, pooling, 
                                    pos_word_embedding, connection)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.pooler_window = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=ROBERTA_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=ROBERTA_CONFIG_FOR_DOC,
    )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, _, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #flatten timeline dimension into batch dimension
        attn_mask_shape = input_ids[:,0,:].size()
        input_ids = torch.flatten(input_ids,0,1)
        token_type_ids = torch.flatten(token_type_ids,0,1)
        attention_mask = torch.flatten(attention_mask,0,1)
        zero_mask = torch.flatten(zero_mask,0,1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, attn_mask_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None: #ttseriotou
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            zero_mask=zero_mask,
            time=time,
            stream_dims = input_shape,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0][:,-1,:,:]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        pooled_window_output = encoder_outputs[1][-1][:,-1,:]
        pooled_output = torch.cat((pooled_output, pooled_window_output), 1)
       
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] 
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class TempoFormerClassification(torch.nn.Module):
    def __init__(self,
        base_model: str,
        hidden_dim: list[int] | tuple[int] | int,
        output_dim: int,
        dropout_rate_ffn: float,
        position_seqlayer: list[int] | tuple[int] | int,
        window: int,
        sequential_pooling: str | None,
        pos_word_embedding: str | None,
        connection: str|None,
        ):

        super(TempoFormerClassification, self).__init__()
        if (base_model=='bert'):
            config = BertConfig.from_pretrained("bert-base-uncased")
            self.tempoformer = SeqBertModel.from_pretrained("bert-base-uncased",
                                position_seqlayer=position_seqlayer,
                                window=window,
                                pooling=sequential_pooling,
                                pos_word_embedding=pos_word_embedding,
                                connection=connection)
        elif (base_model=='roberta'):
            config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")
            self.tempoformer = SeqRobertaModel.from_pretrained("FacebookAI/roberta-base",
                                position_seqlayer=position_seqlayer,
                                window=window,
                                pooling=sequential_pooling,
                                pos_word_embedding=pos_word_embedding,
                                connection=connection)
        else:
            raise ValueError("`model_type` must be one of: 'seqbert', 'seqroberta'")

        #ffn classifier
        self.classifier = FeedforwardNeuralNetModel(
            input_dim=2*config.hidden_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate_ffn,
        )
            
    def forward(self, ids, mask, token_type_ids, zero_mask, time=None):
        _, pooled_output, _ = self.tempoformer(ids, 
                                    attention_mask=mask, 
                                    token_type_ids=token_type_ids, 
                                    zero_mask=zero_mask, 
                                    time=time,
                                    output_hidden_states=True,
                                    return_dict=False)

        logits = self.classifier(pooled_output)
        return logits


class RoTempoFormerClassification(torch.nn.Module):
    def __init__(self,
        base_model: str,
        lstm_hidden_dim: int,
        hidden_dim: list[int] | tuple[int] | int,
        output_dim: int,
        dropout_rate_ffn: float,
        position_seqlayer: list[int] | tuple[int] | int,
        window: int,
        sequential_pooling: str | None,
        pos_word_embedding: str | None,
        connection: str|None,
        ):

        super(RoTempoFormerClassification, self).__init__()
        if (base_model=='bert'):
            self.tempoformer = SeqBertModel.from_pretrained("bert-base-uncased",
                                position_seqlayer=position_seqlayer,
                                window=window,
                                pooling=sequential_pooling,
                                pos_word_embedding=pos_word_embedding,
                                connection=connection)
        elif (base_model=='roberta'):
            self.tempoformer = SeqRobertaModel.from_pretrained("FacebookAI/roberta-base",
                                position_seqlayer=position_seqlayer,
                                window=window,
                                pooling=sequential_pooling,
                                pos_word_embedding=pos_word_embedding,
                                connection=connection)
        else:
            raise ValueError("`model_type` must be one of: 'seqbert', 'seqroberta'")
        #pooler-like layer (without the pooling)
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

        #recurrence
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_dim, num_layers=1)

        #ffn classifier
        self.classifier = FeedforwardNeuralNetModel(
            input_dim=lstm_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate_ffn,
        )
            
    def forward(self, ids, mask, token_type_ids, zero_mask, time=None):

        _, _, cls_outputs = self.tempoformer(ids, 
                                    attention_mask=mask, 
                                    token_type_ids=token_type_ids, 
                                    zero_mask=zero_mask, 
                                    time=time,
                                    output_hidden_states=True,
                                    return_dict=False)

        #flip the padding to fit lstm input requirements
        cls_outputs = torch.flip(cls_outputs, dims=(1,))

        #pooling-like  operation
        cls_outputs = self.dense(cls_outputs)
        cls_outputs = self.activation(cls_outputs)
        
        #pad with lstm using the mask
        seq_lengths = torch.sum(~zero_mask, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        cls_outputs = cls_outputs[perm_idx]

        #lstm
        x_pack = nn.utils.rnn.pack_padded_sequence(cls_outputs, seq_lengths.cpu(), batch_first=True)
        _, (out_h, _) = self.lstm(x_pack)

        #reverse padding/sorting
        #out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        inverse_perm = torch.argsort(perm_idx)
        out_h = torch.squeeze(out_h, dim=0)
        out_h = out_h[inverse_perm]

        out_h = self.classifier(out_h)

        return out_h
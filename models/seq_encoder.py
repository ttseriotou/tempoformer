import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
) 
from transformers.models.bert.modeling_bert import BertLayer, BertOutput, BertIntermediate
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaOutput, RobertaIntermediate
from transformers.pytorch_utils import apply_chunking_to_forward
from models.sequence_mha import GatedConnection
from models.rope_mha import MHARoPE

class LocalStreamEmbedding(nn.Module):
    def __init__(self, config, window, pos_word_embedding):
        super().__init__()
        self.window = window
        self.pos_word_embedding = pos_word_embedding

        if self.pos_word_embedding not in ["sinusoidal", "learnable", None]:
            raise ValueError("`pos_word_embedding` must be one of: 'sinusoidal', 'learnable', None")

        #local positional embeddings
        if (self.pos_word_embedding == 'sinusoidal'):
                #add sunisoidal positional encodings to retrive the sequential aspect of posts
                d_pos_vec = config.hidden_size
                n_position = self.window
                self.position_enc_word = torch.Tensor([
                    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
                    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
                self.position_enc_word[1:, 0::2] = torch.sin(self.position_enc_word[1:, 0::2]) # dim 2i
                self.position_enc_word[1:, 1::2] = torch.cos(self.position_enc_word[1:, 1::2]) # dim 2i+1
        elif (self.pos_word_embedding =='learnable'):
                # initialise absolute position embeddings for the posts
                self.position_enc_word = nn.Embedding(self.window, config.hidden_size)
                # layer norm and dropout after adding the positional embeddings
                self.pos_layernorm_word = nn.LayerNorm(config.hidden_size)
                self.pos_dropout_word = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        ######EMBEDDINGS##########
        #positional embeddings id mapping - local
        if (self.pos_word_embedding == 'sinusoidal'):
            position_emb_lc = self.position_enc_word.unsqueeze(1).unsqueeze(0).to(device)
        elif (self.pos_word_embedding == 'learnable'):
            # obtain position_ids of size [batch, window size]
            position_ids = torch.arange(self.window, device=device).repeat(batch_size, 1)
            position_emb_lc = self.position_enc_word(position_ids)
            position_emb_lc= position_emb_lc.unsqueeze(2)
        
        return self.pos_dropout_word(self.pos_layernorm_word(x+position_emb_lc)) if (self.pos_word_embedding == 'learnable') else x+position_emb_lc

def copy_weights(new, existing):
        target = dict(new.named_parameters())
        source = dict(existing.named_parameters())
        for part in source.keys():
            target[part].data.copy_(source[part].data) 

class SeqLayer(nn.Module):
    def __init__(self, config, pretrained_layer, window, pooling, pos_word_embedding, connection):
        super().__init__()  

        self.pooling = pooling
        self.pos_word_embedding = pos_word_embedding
        self.connection = connection

        if self.pos_word_embedding is not None:
            self.localembedding = LocalStreamEmbedding(config, window, pos_word_embedding)

        self.transformer_layer = BertLayer(config)
        copy_weights(self.transformer_layer, pretrained_layer)

        if self.pooling == 'mha_rotary':
            self.mha_cls = MHARoPE(embed_dim= config.hidden_size, 
                            num_heads=config.num_attention_heads, 
                            dropout=config.attention_probs_dropout_prob, 
                            max_seq_len=window)
        else:
            self.mha_cls = nn.MultiheadAttention(embed_dim= config.hidden_size, 
                            num_heads=config.num_attention_heads, 
                            dropout=config.attention_probs_dropout_prob, 
                            bias=True, 
                            batch_first=True)

        #fusion of local and global states
        if (self.connection == 'gatednorm'):
            self.gate = GatedConnection(config.hidden_size)
            self.gated_layernorm = nn.LayerNorm(config.hidden_size)
        elif (self.connection == 'layernorm'):
            self.linear_glb = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout_glb = nn.Dropout(config.hidden_dropout_prob)
            self.layernorm_glb = nn.LayerNorm(config.hidden_size)

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        stream_dims: Optional[torch.IntTensor] = None,
        layer_head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False):

        if self.pos_word_embedding is not None:
            hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
            hidden_states = self.localembedding(hidden_states)
            hidden_states = torch.flatten(hidden_states,0,1)

        #remove if statement for gradient_checkpointing and training
        layer_outputs = self.transformer_layer(
                        hidden_states[~zero_mask],
                        attention_mask[~zero_mask],
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
        hidden_states[~zero_mask] = layer_outputs[0] #dims: [batch, window, seq_len, embd_dim]
        hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
        mask_attn_avg = torch.unflatten(zero_mask, 0, (stream_dims[0], stream_dims[1]))

        h_emb = hidden_states[:,:,0,:] #take cls tokens

        #global layer - flipped window
        h_emb = torch.flip(h_emb, dims=(1,))
        mask_attn_avg = torch.flip(mask_attn_avg, dims=(1,)) 
        if time != None:
            dt_tr = time - torch.max(time,axis=1).values.unsqueeze(1)
            dt_tr[time==0] = 1
            dt_tr[dt_tr==0] = 1 
            dt_tr = torch.abs(dt_tr)
            dt_tr = torch.log(dt_tr) 
            dt_tr = dt_tr.flip(1) 
        else:
            dt_tr=None
        if self.pooling == 'mha_rotary':
            h_emb = self.mha_cls(x=h_emb, key_padding_mask=mask_attn_avg, dt=dt_tr)
        else:
            h_emb, _ = self.mha_cls(query=h_emb, key=h_emb, value=h_emb, key_padding_mask=mask_attn_avg)
          
        #gating of local and global representations 
        h_emb = torch.flip(h_emb, dims=(1,))
        if (self.connection == 'gatednorm'):
            h_G = self.gate(hidden_states[:,:,0,:], h_emb) 
            hidden_states_cls = self.gated_layernorm(h_G)
        elif (self.connection == 'layernorm'):
            h_G = self.linear_glb(h_emb)
            h_G = self.dropout_glb(h_G)
            hidden_states_cls = self.layernorm_glb(h_G + hidden_states[:,:,0,:])
        else:
            hidden_states_cls = h_emb 
        
        return hidden_states_cls, hidden_states

class SeqBertEncoder(nn.Module):
    def __init__(self, config, position_seqlayer, window, pooling, pos_word_embedding, connection):
        super().__init__()
        self.config = config
        #Note to self: we should follow the same naming if we want weight initialization for modified layers. Explore more here:
        #https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py _load_pretrained_model()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.intermediate_hist = BertIntermediate(config)
        self.output_hist = BertOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.gradient_checkpointing = False

        self.seq_layer = nn.ModuleList([SeqLayer(config, self.layer[p], window, pooling, pos_word_embedding, connection) if (i==(len(position_seqlayer)-1)) else SeqLayer(config, self.layer[p], window, pooling, pos_word_embedding, None) for i,p in enumerate(position_seqlayer)])
        self.position_seqlayer = position_seqlayer

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate_hist(attention_output)
        layer_output = self.output_hist(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        stream_dims: Optional[torch.IntTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        ind_seqlayer = 0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1])),)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if (i in self.position_seqlayer):
                hidden_states_cls, hs = self.seq_layer[ind_seqlayer](hidden_states,
                        attention_mask,
                        zero_mask,
                        time,
                        stream_dims,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions)
                ind_seqlayer+=1

                if i!=(len(self.layer)-1):
                    hidden_states = hs.clone()#all_hidden_states[-1] 
                    hidden_states[:,:,0,:] = hidden_states_cls
                    hidden_states = torch.flatten(hidden_states,0,1)
            else:
                layer_outputs = layer_module(
                    hidden_states[~zero_mask],
                    attention_mask[~zero_mask],
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )      
                hidden_states[~zero_mask] = layer_outputs[0]
                last_hidden_state = hidden_states #here
                if i==(len(self.layer)-1):
                    hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],) 
        ##############################################################################
        ##############################################################################
        #unflatten states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) + (hidden_states_cls,) #HERE
            #all_hidden_states = (hidden_states_cls,)

        if last_hidden_state.dim()<4 : 
            last_hidden_state = torch.unflatten(last_hidden_state, 0, (stream_dims[0], stream_dims[1])) #HERE

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state, #HERE
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

class SeqLayer(nn.Module):
    def __init__(self, config, pretrained_layer, window, pooling, pos_word_embedding, connection):
        super().__init__()  

        self.pooling = pooling
        self.pos_word_embedding = pos_word_embedding
        self.connection = connection

        if self.pos_word_embedding is not None:
            self.localembedding = LocalStreamEmbedding(config, window, pos_word_embedding)

        self.transformer_layer = BertLayer(config)
        copy_weights(self.transformer_layer, pretrained_layer)

        if self.pooling == 'mha_rotary':
            self.mha_cls = MHARoPE(embed_dim= config.hidden_size, 
                            num_heads=config.num_attention_heads, 
                            dropout=config.attention_probs_dropout_prob, 
                            max_seq_len=window)
        else:
            self.mha_cls = nn.MultiheadAttention(embed_dim= config.hidden_size, 
                            num_heads=config.num_attention_heads, 
                            dropout=config.attention_probs_dropout_prob, 
                            bias=True, 
                            batch_first=True)

        #fusion of local and global states
        if (self.connection == 'gatednorm'):
            self.gate = GatedConnection(config.hidden_size)
            self.gated_layernorm = nn.LayerNorm(config.hidden_size)
        elif (self.connection == 'layernorm'):
            self.linear_glb = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout_glb = nn.Dropout(config.hidden_dropout_prob)
            self.layernorm_glb = nn.LayerNorm(config.hidden_size)

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        stream_dims: Optional[torch.IntTensor] = None,
        layer_head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False):

        if self.pos_word_embedding is not None:
            hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
            hidden_states = self.localembedding(hidden_states)
            hidden_states = torch.flatten(hidden_states,0,1)

        #remove if statement for gradient_checkpointing and training
        layer_outputs = self.transformer_layer(
                        hidden_states[~zero_mask],
                        attention_mask[~zero_mask],
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
        hidden_states[~zero_mask] = layer_outputs[0] #dims: [batch, window, seq_len, embd_dim]
        hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
        mask_attn_avg = torch.unflatten(zero_mask, 0, (stream_dims[0], stream_dims[1]))

        h_emb = hidden_states[:,:,0,:] #take cls tokens

        #global layer - flipped window
        h_emb = torch.flip(h_emb, dims=(1,))
        mask_attn_avg = torch.flip(mask_attn_avg, dims=(1,)) 
        if time != None:
            dt_tr = time - torch.max(time,axis=1).values.unsqueeze(1)
            dt_tr[time==0] = 1
            dt_tr[dt_tr==0] = 1 
            dt_tr = torch.abs(dt_tr)
            dt_tr = torch.log(dt_tr) 
            dt_tr = dt_tr.flip(1) 
        else:
            dt_tr=None
        if self.pooling == 'mha_rotary':
            h_emb = self.mha_cls(x=h_emb, key_padding_mask=mask_attn_avg, dt=dt_tr)
        else:
            h_emb, _ = self.mha_cls(query=h_emb, key=h_emb, value=h_emb, key_padding_mask=mask_attn_avg)
          
        #gating of local and global representations
        h_emb = torch.flip(h_emb, dims=(1,))
        if (self.connection == 'gatednorm'):
            h_G = self.gate(hidden_states[:,:,0,:], h_emb) 
            hidden_states_cls = self.gated_layernorm(h_G)
        elif (self.connection == 'layernorm'):
            h_G = self.linear_glb(h_emb)
            h_G = self.dropout_glb(h_G)
            hidden_states_cls = self.layernorm_glb(h_G + hidden_states[:,:,0,:])
        else:
            hidden_states_cls = h_emb 
        return hidden_states_cls, hidden_states


class SeqRobertaEncoder(nn.Module):
    def __init__(self, config, position_seqlayer, window, pooling, pos_word_embedding, connection):
        super().__init__()
        self.config = config
        #Note to self: we should follow the same naming if we want weight initialization for modified layers. Explore more here:
        #https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py _load_pretrained_model()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

        self.intermediate_hist = RobertaIntermediate(config)
        self.output_hist = RobertaOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.gradient_checkpointing = False

        self.seq_layer = nn.ModuleList([SeqLayer(config, self.layer[p], window, pooling, pos_word_embedding, connection) if (i==(len(position_seqlayer)-1)) else SeqLayer(config, self.layer[p], window, pooling, pos_word_embedding, None) for i,p in enumerate(position_seqlayer)])
        self.position_seqlayer = position_seqlayer

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate_hist(attention_output)
        layer_output = self.output_hist(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        zero_mask: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        stream_dims: Optional[torch.IntTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        ind_seqlayer = 0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1])),)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if (i in self.position_seqlayer):
                hidden_states_cls, hs = self.seq_layer[ind_seqlayer](hidden_states,
                        attention_mask,
                        zero_mask,
                        time,
                        stream_dims,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions)
                ind_seqlayer+=1

                if i!=(len(self.layer)-1):
                    hidden_states = hs.clone()
                    hidden_states[:,:,0,:] = hidden_states_cls
                    hidden_states = torch.flatten(hidden_states,0,1)
            else:
                layer_outputs = layer_module(
                    hidden_states[~zero_mask],
                    attention_mask[~zero_mask],
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )      
                hidden_states[~zero_mask] = layer_outputs[0]
                last_hidden_state = hidden_states 
                if i==(len(self.layer)-1):
                    hidden_states = torch.unflatten(hidden_states, 0, (stream_dims[0], stream_dims[1]))
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],) 
        ##############################################################################
        ##############################################################################
        #unflatten states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) + (hidden_states_cls,) 
            #all_hidden_states = (hidden_states_cls,)

        if last_hidden_state.dim()<4 : 
            last_hidden_state = torch.unflatten(last_hidden_state, 0, (stream_dims[0], stream_dims[1])) 

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state, 
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
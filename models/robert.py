import torch.nn as nn 
import torch
from transformers import BertModel
from models.ffn import FeedforwardNeuralNetModel


class RoBERT(nn.Module):
    """
    Recurrence over BERT (RoBERT).
    
    Non-timensitive, hierarchical transformer model.
    """
    def __init__(self, 
                output_dim: int,
                dropout_rate_ffn: float,
                window: int,
                lstm_hidden_dim: int,
                hidden_dim_ffn: int):
        
        super(RoBERT, self).__init__()        
        # Initialization
        self.dropout_rate_ffn=dropout_rate_ffn
        self.output_dim=output_dim
        self.window=window
        self.lstm_hidden_dim=lstm_hidden_dim
        self.hidden_dim_ffn = hidden_dim_ffn
        
        # Layers
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.lstm_hidden_dim, num_layers=1)

        self.ffnetwork = FeedforwardNeuralNetModel(
            input_dim=self.lstm_hidden_dim,
            hidden_dim=self.hidden_dim_ffn,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate_ffn,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, zero_mask):

        #device
        device = input_ids.device

        #flatten timeline dimension into batch dimension
        input_shape = input_ids.size()

        input_ids = torch.flatten(input_ids,0,1)
        token_type_ids = torch.flatten(token_type_ids,0,1)
        attention_mask = torch.flatten(attention_mask,0,1)
        zero_mask = torch.flatten(zero_mask,0,1)

        _, pooled_output = self.bert(input_ids[~zero_mask], 
                                    attention_mask=attention_mask[~zero_mask], 
                                    token_type_ids=token_type_ids[~zero_mask], 
                                    return_dict=False)
        hidden_states = torch.zeros((input_shape[0]*input_shape[1], pooled_output.shape[-1]), device=device) #maintain a zero tensor with the original size

        #prepare input for lstm
        hidden_states[~zero_mask] = pooled_output #dims: [batch x window, embd_dim]

        #unflatten the hidden states and the zero mask
        hidden_states = torch.unflatten(hidden_states, 0, (input_shape[0],input_shape[1])) #dims: [batch, window, embd_dim]
        mask_attn_avg = torch.unflatten(zero_mask, 0, (input_shape[0],input_shape[1])) #dims: [batch, window]

        #flip the padding to fit lstm input requirements
        hidden_states = torch.flip(hidden_states, dims=(1,))

        #pad with lstm using the mask
        seq_lengths = torch.sum(~mask_attn_avg, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        hidden_states = hidden_states[perm_idx]

        #lstm
        x_pack = nn.utils.rnn.pack_padded_sequence(hidden_states, seq_lengths.cpu(), batch_first=True)
        out, (out_h, _) = self.lstm(x_pack)

        #reverse padding/sorting
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        inverse_perm = torch.argsort(perm_idx)
        out_h = torch.squeeze(out_h, dim=0)
        out_h = out_h[inverse_perm]
       
        #ffn as in original paper with ReLU
        logits = self.ffnetwork(out_h)

        return logits
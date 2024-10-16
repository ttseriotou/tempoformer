from __future__ import annotations

import torch
import torch.nn as nn

from models.feature_concatenation import FeatureConcatenation
from models.ffn import FeedforwardNeuralNetModel
from models.swnu import SWNU


class SeqSigNet(nn.Module):
    """
    BiLSTM of Deep Signature Neural Network Units for classification.
    """

    def __init__(
        self,
        input_channels: int,
        num_features: int,
        embedding_dim: int,
        log_signature: bool,
        sig_depth: int,
        pooling: str,
        hidden_dim_swnu: list[int] | int,
        hidden_dim_lstm: int,
        hidden_dim_ffn: list[int] | int,
        output_dim: int,
        dropout_rate: float,
        reverse_path: bool = False,
        output_channels: int | None = None,
        augmentation_type: str = "Conv1d",
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
        comb_method: str = "concatenation",
    ):
        """
        SeqSigNet network for classification.

        Input data will have the size: [batch size, window size (w),
        all embedding dimensions (history + time + post), unit size (n)]
        Note: unit sizes will be in reverse chronological order, starting
        from the more recent and ending with the one further back in time.

        Parameters
        ----------
        input_channels : int
            Dimension of the embeddings in the path that will be passed in.
        num_features : int
            Number of time features to add to FFN input. If none, set to zero.
        embedding_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        pooling: str
            Pooling operation to apply in SWNU to obtain history representation.
            Options are:
                - "signature": apply signature on the LSTM units at the end
                  to obtain the final history representation
                - "lstm": take the final (non-padded) LSTM unit as the final
                  history representation
        hidden_dim_swnu : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
        hidden_dim_lstm : int
            Dimensions of the hidden layers in the final BiLSTM applied to the output
            of the SWNU units.
        hidden_dim_ffn : list[int] | int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        reverse_path : bool, optional
            Whether or not to reverse the path before passing it through the
            signature layers, by default False.
        output_channels : int | None, optional
            Requested dimension of the embeddings after convolution layer.
            If None, will be set to the last item in `hidden_dim`, by default None.
        augmentation_type : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - "signatory": passes path through `Augment` layer from `signatory` package.
        hidden_dim_aug : list[int] | int | None
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type='signatory'`, by default None.
        BiLSTM : bool, optional
            Whether or not a birectional LSTM is used in the SWNUs,
            by default False (unidirectional LSTM is used in this case).
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature
              and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature
              and embedding vector
            - scaled_concatenation: concatenation of single value scaled path
              signature and embedding vector
        """
        super().__init__()

        if pooling not in ["signature", "lstm"]:
            raise ValueError(
                "`pooling` must be 'signature' or 'lstm'. " f"Got {pooling} instead."
            )

        self.swnu = SWNU(
            input_channels=input_channels,
            output_channels=output_channels,
            log_signature=log_signature,
            sig_depth=sig_depth,
            hidden_dim=hidden_dim_swnu,
            pooling=pooling,
            reverse_path=reverse_path,
            BiLSTM=BiLSTM,
            augmentation_type=augmentation_type,
            hidden_dim_aug=hidden_dim_aug,
        )

        # BiLSTM that processes the outputs from SWNUs for each window
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm_sig = nn.LSTM(
            input_size=self.swnu.swlstm.output_dim,
            hidden_size=self.hidden_dim_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # determining how to concatenate features to the SWNU features
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.comb_method = comb_method
        self.feature_concat = FeatureConcatenation(
            input_dim=self.hidden_dim_lstm,
            num_features=self.num_features,
            embedding_dim=self.embedding_dim,
            comb_method=self.comb_method,
        )

        # FFN for classification
        # make sure hidden_dim_ffn a list of integers
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn

        self.ffn = FeedforwardNeuralNetModel(
            input_dim=self.feature_concat.output_dim,
            hidden_dim=self.hidden_dim_ffn,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, path: torch.Tensor, features: torch.Tensor | None = None):
        # path has dimensions [batch, units, history, channels]
        # features has dimensions [batch, num_features+embedding_dim]
        # SWNU for each history window by flattening and unflattening the path
        # first flatten the path to a three-dimensional tensor of
        # dimensions [batch*units, history, channels]
        out_flat = path.flatten(0, 1)
        # apply SWNU to out_flat
        out = self.swnu(out_flat)
        # unflatten out to have dimensions [batch, units, hidden_dim]
        out = out.unflatten(0, (path.shape[0], path.shape[1]))

        # order sequences based on sequence length of input
        # for each item in the batch dimension, find the number of non-zero windows
        # (i.e. the number of windows that are not fully padded with zeros)
        seq_lengths = torch.sum(torch.sum(path, (2, 3)) != 0, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        out = out[perm_idx]
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, seq_lengths.cpu(), batch_first=True
        )

        # BiLSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig(out)
        out = out[-1, :, :] + out[-2, :, :]

        # reverse sequence padding
        inverse_perm = torch.argsort(perm_idx)
        out = out[inverse_perm]

        # combine with features provided
        out = self.feature_concat(out, features)

        # FFN
        out = self.ffn(out.float())

        return out

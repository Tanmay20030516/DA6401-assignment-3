import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # TODO: Task 2.1 - Implement sequential sinusoidal positional encodings
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # TODO: Task 2.1 - Add positional encoding to input x
        raise NotImplementedError

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        # TODO: Task 1.2 - Initializations for Multi-Head Attention
        # Note: You cannot use torch.nn.MultiheadAttention here!
        pass

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        TODO: Task 1.1 + 1.2 - Implement Scaled Dot-Product and Multi-Head Attention
        """
        raise NotImplementedError

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        # TODO: Task 2.3 - Implement the two-layer linear transformation with ReLU
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Task 2.3 - FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        """
        raise NotImplementedError

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        # TODO: Initialize components for the Encoder Layer (Self-Attention, FFN, Add & Norm)
        pass

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        # TODO: Implement the forward pass through the encoder layer
        raise NotImplementedError

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        # TODO: Initialize components for the Decoder Layer (Self-Attention, Cross-Attention, FFN, Add & Norm)
        pass

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # TODO: Implement the forward pass through the decoder layer
        raise NotImplementedError

class Encoder(nn.Module):
    def __init__(self, layer, N: int):
        super(Encoder, self).__init__()
        # TODO: Stack N encoder layers
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # TODO: Pass x through the stacked layers
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, layer, N: int):
        super(Decoder, self).__init__()
        # TODO: Stack N decoder layers
        pass

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # TODO: Pass x through the stacked layers
        raise NotImplementedError

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        # TODO: Combine Encoder, Decoder, Generator, Positional Ecodings and Embeddings
        pass

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # TODO: Forward pass for the entire transformer architecture
        raise NotImplementedError

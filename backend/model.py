"""
GPT-Style Transformer Model
===========================
A simplified but functional transformer architecture for text generation.
This implementation is educational and designed to be understood by beginners.

Architecture Overview:
- Token Embedding: Converts words/characters to vectors
- Positional Encoding: Adds position information to embeddings
- Transformer Blocks: Multiple layers of self-attention and feed-forward networks
- Output Layer: Predicts next token in sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism

    This is the core of the transformer. It allows the model to focus on
    different parts of the input sequence simultaneously.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Query, Key, Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Apply linear transformations and split into multiple heads
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (prevents looking at future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        # Shape: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_linear(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    A simple two-layer neural network applied to each position independently.
    Typically expands to 4x the model dimension then contracts back.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand, activate with GELU, then contract
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation (smoother than ReLU)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Block

    Combines self-attention and feed-forward layers with residual connections
    and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        # Pre-LN architecture (normalize before attention)
        attention_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attention_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)

        return x


class GPTModel(nn.Module):
    """
    Complete GPT-Style Transformer Model

    Parameters:
    - vocab_size: Number of unique tokens in vocabulary
    - d_model: Dimension of model embeddings (default: 256)
    - num_heads: Number of attention heads (default: 8)
    - num_layers: Number of transformer blocks (default: 6)
    - d_ff: Dimension of feed-forward layer (default: 1024)
    - max_seq_len: Maximum sequence length (default: 512)
    - dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6,
                 d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Forward pass through the model

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        # Combine token and position embeddings
        x = self.dropout(token_embeds + position_embeds)

        # Create causal mask (lower triangular matrix)
        # This prevents positions from attending to future positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_final(x)

        # Project to vocabulary size
        logits = self.output_projection(x)

        return logits

    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text by predicting one token at a time

        Args:
            input_ids: Starting tokens (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            generated_ids: Tensor containing original + generated tokens
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop sequence if it exceeds max length
                input_crop = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

                # Get predictions
                logits = self.forward(input_crop)

                # Focus on last time step
                logits = logits[:, -1, :] / temperature

                # Optional: apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_parameters(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test of the model
    print("Testing GPT Model...")

    vocab_size = 1000
    model = GPTModel(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=4)

    print(f"Model initialized with {model.get_num_parameters():,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    generated = model.generate(test_input[:1], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")

    print("✓ Model test successful!")

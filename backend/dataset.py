"""
Dataset Management for Transformer Training
===========================================
Handles loading, preprocessing, and batching text data for training.
Supports multiple sources: files, URLs, and raw text.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import requests
import re
from typing import List, Optional
from pathlib import Path
import json


class TextTokenizer:
    """
    Simple character-level tokenizer

    For beginners, we start with character-level tokenization which is simpler
    than word-level or subword tokenization (like BPE).
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, text: str):
        """Build vocabulary from text"""
        # Get unique characters
        chars = sorted(list(set(text)))

        # Add special tokens
        self.char2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2char = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}

        # Add regular characters
        for i, char in enumerate(chars, start=4):
            self.char2idx[char] = i
            self.idx2char[i] = char

        self.vocab_size = len(self.char2idx)
        print(f"Vocabulary built with {self.vocab_size} characters")

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in token_ids])

    def save(self, filepath: str):
        """Save tokenizer to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {int(k): v for k, v in self.idx2char.items()},
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab_size = data['vocab_size']


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data

    Splits text into sequences of fixed length for training.
    """

    def __init__(self, text: str, tokenizer: TextTokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Encode entire text
        self.tokens = tokenizer.encode(text)

        # Calculate number of sequences we can create
        self.num_sequences = max(1, len(self.tokens) - seq_length)

        print(f"Dataset created with {len(self.tokens):,} tokens and {self.num_sequences:,} sequences")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get a training sample

        Returns:
            input_ids: Tensor of shape (seq_length,) containing input tokens
            target_ids: Tensor of shape (seq_length,) containing target tokens (shifted by 1)
        """
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.seq_length + 1]

        # Pad if necessary
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [self.tokenizer.char2idx['<PAD>']] * (self.seq_length + 1 - len(chunk))

        # Input and target (target is input shifted by 1)
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


class DatasetManager:
    """
    High-level interface for managing datasets

    Handles loading from multiple sources and creating DataLoaders.
    """

    def __init__(self, seq_length: int = 128):
        self.seq_length = seq_length
        self.tokenizer = TextTokenizer()
        self.raw_texts = []
        self.combined_text = ""

    def add_text(self, text: str, source: str = "manual"):
        """Add raw text to the dataset"""
        # Clean the text
        text = self._clean_text(text)
        self.raw_texts.append({
            'text': text,
            'source': source,
            'length': len(text)
        })
        print(f"Added text from {source}: {len(text):,} characters")

    def add_file(self, filepath: str):
        """Load text from a file"""
        try:
            path = Path(filepath)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            self.add_text(text, source=f"file:{path.name}")
            return True
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return False

    def add_url(self, url: str):
        """Fetch and add text from a URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Extract text from HTML (simple approach)
            text = response.text

            # Remove HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)

            # Decode HTML entities
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

            self.add_text(text, source=f"url:{url[:50]}")
            return True
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def prepare(self):
        """Prepare the dataset for training"""
        if not self.raw_texts:
            raise ValueError("No text data added! Use add_text(), add_file(), or add_url() first.")

        # Combine all texts
        self.combined_text = ' '.join([item['text'] for item in self.raw_texts])

        print(f"\nDataset Summary:")
        print(f"  Total sources: {len(self.raw_texts)}")
        print(f"  Total characters: {len(self.combined_text):,}")

        # Build vocabulary
        self.tokenizer.build_vocab(self.combined_text)

        # Create dataset
        self.dataset = TextDataset(self.combined_text, self.tokenizer, self.seq_length)

        return self.dataset

    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader"""
        if not hasattr(self, 'dataset'):
            self.prepare()

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )

    def get_sample_text(self, num_chars: int = 200) -> str:
        """Get a sample from the dataset for preview"""
        return self.combined_text[:num_chars] + "..." if len(self.combined_text) > num_chars else self.combined_text

    def save_tokenizer(self, filepath: str):
        """Save the tokenizer for later use"""
        self.tokenizer.save(filepath)

    def load_tokenizer(self, filepath: str):
        """Load a previously saved tokenizer"""
        self.tokenizer.load(filepath)


# Default sample text for quick testing
DEFAULT_SAMPLE_TEXT = """
The transformer architecture has revolutionized natural language processing.
It uses self-attention mechanisms to process sequences of data in parallel,
making it much faster than recurrent neural networks. The key innovation is
the attention mechanism, which allows the model to focus on different parts
of the input when making predictions. This architecture powers models like
GPT, BERT, and many others that have achieved state-of-the-art results on
various NLP tasks. Training a transformer from scratch is an excellent way
to understand how modern AI systems work.
"""


if __name__ == "__main__":
    # Test the dataset manager
    print("Testing Dataset Manager...")

    manager = DatasetManager(seq_length=64)

    # Add sample text
    manager.add_text(DEFAULT_SAMPLE_TEXT, source="test")

    # Prepare dataset
    dataset = manager.prepare()

    # Create dataloader
    dataloader = manager.create_dataloader(batch_size=4)

    # Test one batch
    input_ids, target_ids = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Target: {target_ids.shape}")

    # Test encoding/decoding
    sample_text = "Hello, World!"
    encoded = manager.tokenizer.encode(sample_text)
    decoded = manager.tokenizer.decode(encoded)
    print(f"\nEncoding test:")
    print(f"  Original: {sample_text}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")

    print("\n✓ Dataset test successful!")

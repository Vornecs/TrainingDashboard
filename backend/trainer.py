"""
Training Pipeline for Transformer Model
=======================================
Handles training loop, checkpointing, and metrics tracking.
Provides real-time monitoring capabilities for the web interface.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from pathlib import Path
from typing import Optional, Callable
import math


class Trainer:
    """
    Manages the training process for the transformer model

    Features:
    - Automatic checkpointing
    - Learning rate scheduling
    - Gradient clipping
    - Training metrics tracking
    - Callback support for real-time updates
    """

    def __init__(
        self,
        model,
        train_dataloader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 10,
        device: str = 'auto',
        checkpoint_dir: str = './checkpoints',
        grad_clip: float = 1.0
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip

        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Setup optimizer (AdamW is standard for transformers)
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)  # Common settings for transformers
        )

        # Learning rate scheduler (cosine annealing)
        total_steps = len(train_dataloader) * max_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)

        # Loss function (cross-entropy for language modeling)
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_stats = []
        self.is_training = False
        self.should_stop = False

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks for real-time updates
        self.step_callbacks = []
        self.epoch_callbacks = []

    def add_step_callback(self, callback: Callable):
        """Add a callback function called after each training step"""
        self.step_callbacks.append(callback)

    def add_epoch_callback(self, callback: Callable):
        """Add a callback function called after each epoch"""
        self.epoch_callbacks.append(callback)

    def train_step(self, input_ids, target_ids):
        """
        Perform a single training step

        Returns:
            loss: The loss value for this step
        """
        self.model.train()

        # Move data to device
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        logits = self.model(input_ids)

        # Calculate loss
        # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.criterion(
            logits.view(batch_size * seq_len, vocab_size),
            target_ids.view(batch_size * seq_len)
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def calculate_perplexity(self, loss):
        """
        Calculate perplexity from loss

        Perplexity is a common metric for language models.
        Lower is better. It roughly represents how "confused" the model is.
        """
        return math.exp(loss) if loss < 20 else float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = len(self.train_dataloader)

        for batch_idx, (input_ids, target_ids) in enumerate(self.train_dataloader):
            if self.should_stop:
                break

            # Training step
            step_start_time = time.time()
            loss = self.train_step(input_ids, target_ids)
            step_time = time.time() - step_start_time

            total_loss += loss
            self.global_step += 1

            # Calculate metrics
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = self.calculate_perplexity(avg_loss)
            current_lr = self.scheduler.get_last_lr()[0]

            # Create step info
            step_info = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'batch': batch_idx + 1,
                'total_batches': num_batches,
                'loss': loss,
                'avg_loss': avg_loss,
                'perplexity': perplexity,
                'learning_rate': current_lr,
                'step_time': step_time,
                'tokens_per_sec': (input_ids.shape[0] * input_ids.shape[1]) / step_time
            }

            # Call step callbacks
            for callback in self.step_callbacks:
                try:
                    callback(step_info)
                except Exception as e:
                    print(f"Error in step callback: {e}")

            # Print progress every 10 steps
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {self.current_epoch} | Step {batch_idx + 1}/{num_batches} | "
                      f"Loss: {loss:.4f} | Perplexity: {perplexity:.2f} | "
                      f"LR: {current_lr:.6f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / num_batches
        epoch_perplexity = self.calculate_perplexity(avg_epoch_loss)

        epoch_info = {
            'epoch': self.current_epoch,
            'avg_loss': avg_epoch_loss,
            'perplexity': epoch_perplexity,
            'epoch_time': epoch_time,
            'global_step': self.global_step
        }

        self.training_stats.append(epoch_info)

        # Call epoch callbacks
        for callback in self.epoch_callbacks:
            try:
                callback(epoch_info)
            except Exception as e:
                print(f"Error in epoch callback: {e}")

        print(f"\nEpoch {self.current_epoch} completed in {epoch_time:.2f}s")
        print(f"Average Loss: {avg_epoch_loss:.4f} | Perplexity: {epoch_perplexity:.2f}\n")

        return epoch_info

    def train(self):
        """Run the complete training loop"""
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Total parameters: {self.model.get_num_parameters():,}")
        print(f"Device: {self.device}\n")

        self.is_training = True
        self.should_stop = False
        training_start_time = time.time()

        try:
            for epoch in range(self.max_epochs):
                if self.should_stop:
                    print("Training stopped by user")
                    break

                self.current_epoch = epoch
                self.train_epoch()

                # Save checkpoint after each epoch
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Training completed
            total_time = time.time() - training_start_time
            print(f"\n{'='*50}")
            print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"Total steps: {self.global_step}")
            print(f"Final loss: {self.training_stats[-1]['avg_loss']:.4f}")
            print(f"{'='*50}\n")

            # Save final model
            self.save_checkpoint("final_model.pt")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_checkpoint("interrupted_checkpoint.pt")

        finally:
            self.is_training = False

    def stop_training(self):
        """Signal the training loop to stop"""
        self.should_stop = True

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_len': self.model.max_seq_len,
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save training stats as JSON for easy access
        stats_path = self.checkpoint_dir / f"{filename.replace('.pt', '')}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_stats = checkpoint['training_stats']

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def generate_sample(self, prompt: str, tokenizer, max_length: int = 100, temperature: float = 0.8):
        """
        Generate text from the model

        Args:
            prompt: Starting text
            tokenizer: Tokenizer to encode/decode text
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        self.model.eval()

        with torch.no_grad():
            # Encode prompt
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(self.device)

            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=50
            )

            # Decode
            generated_text = tokenizer.decode(output_ids[0].cpu().tolist())

        return generated_text


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Import this module and use the Trainer class to train your model.")

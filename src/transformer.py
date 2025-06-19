# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer model."""

import dataclasses
import enum
import functools

import haiku as hk
import jax
import jax.nn as jnn
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict, Optional
import chess

from searchless_chess.src import constants


class PositionalEncodings(enum.Enum):
  SINUSOID = enum.auto()
  LEARNED = enum.auto()


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # The random seed for parameter initialization.
  seed: int = 1
  # The input vocabulary size.
  vocab_size: int
  # The output size (by default equal to the vocabulary size).
  output_size: int | None = None
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 4
  # The number of heads per layer.
  num_heads: int = 8
  # Whether to use a causal mask or not.
  use_causal_mask: bool = True
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # Positional encodings to use.
  pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID
  # Maximum sequence length, useful for the LEARNED positional encodings.
  max_sequence_length: int | None = None
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Whether to apply QK normalization trick in attention layer.
  apply_qk_layernorm: bool = False
  # Whether to apply post LN after attention + MLP blocks
  apply_post_ln: bool = True
  # Whether to use Smolgen for dynamic attention biases. IMPORTANT: It is using Smolgen by default!!
  use_smolgen: bool = True
  # Compression dimension for Smolgen position summary.
  smolgen_compress_dim: int = 32
  # Position summary dimension for Smolgen.
  smolgen_summary_dim: int = 256

  gnn_hidden_dim: int = 128
  gnn_num_layers: int = 3
  gnn_message_passing_steps: int = 2

  def __post_init__(self):
    if self.output_size is None:
      self.output_size = self.vocab_size

  
class PRGModule(hk.Module):
    """Haiku PRG with compressed pair embeddings"""
    def __init__(self, num_heads, compress_dim=32, prg_hidden_dim=64, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.compress_dim = compress_dim  # Smaller than input dim (d)
        self.prg_hidden_dim = prg_hidden_dim

    def __call__(self, x):
        batch_size, seq_len, d = x.shape
        
        # 1. Compress input tokens (d -> compress_dim)
        x_compressed = hk.Linear(self.compress_dim)(x)  # [B, seq_len, compress_dim]
        
        # 2. Create all compressed token pairs
        token_i = jnp.tile(x_compressed[:, :, None, :], (1, 1, seq_len, 1))  # [B, seq_len, seq_len, compress_dim]
        token_j = jnp.tile(x_compressed[:, None, :, :], (1, seq_len, 1, 1))   # [B, seq_len, seq_len, compress_dim]
        pair_emb = jnp.concatenate([token_i, token_j], axis=-1)  # [B, seq_len, seq_len, 2*compress_dim]

        # 3. Process pairs with MLP
        prg_bias = hk.Sequential([
            hk.Linear(self.prg_hidden_dim),
            jax.nn.gelu,
            hk.Linear(self.num_heads)  # [B, seq_len, seq_len, num_heads]
        ])(pair_emb)

        return jnp.transpose(prg_bias, (0, 3, 1, 2))  # [B, num_heads, seq_len, seq_len]

class PRGModule2(hk.Module):
    """Input-dependent PRG with piece/color awareness with precomputed spatial features."""
    def __init__(self, num_heads=8, seq_len=79, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.seq_len = seq_len
        
        # Precompute all spatial features [seq_len, seq_len, 2]
        idx = jnp.arange(seq_len)
        is_board = idx < 64
        x_coord = jnp.where(is_board, idx // 8, 0)
        y_coord = jnp.where(is_board, idx % 8, 0)
        
        dx = x_coord[:, None] - x_coord[None, :]  # [seq_len, seq_len]
        dy = y_coord[:, None] - y_coord[None, :]  # [seq_len, seq_len]
        self.spatial_feats = jnp.stack([dx, dy], axis=-1)  # [seq_len, seq_len, 2]
        
        # Projections for input-dependent features
        self.piece_proj = hk.Linear(6, name="piece_proj")
        self.color_proj = hk.Linear(1, name="color_proj")
        
        # Spatial MLP (now operates on [..., 15] features)
        self.spatial_mlp = hk.Sequential([
            hk.Linear(32), jax.nn.gelu,
            hk.Linear(num_heads)
        ])

    def __call__(self, x):
        B, N, d = x.shape
        assert N == self.seq_len, f"Expected seq_len {self.seq_len}, got {N}"
        
        # Get input-dependent features
        piece_emb = self.piece_proj(x)  # [B, N, 6]
        color_emb = self.color_proj(x)  # [B, N, 1]
        same_color = (color_emb[:, :, None] == color_emb[:, None, :])  # [B, N, N, 1]
        
        # Combine with precomputed spatial features [B, N, N, 15]
        pairwise_feats = jnp.concatenate([
            self.spatial_feats[None, ...].repeat(B, axis=0),  # [B, N, N, 2]
            piece_emb[:, :, None, :].repeat(N, axis=2),       # [B, N, N, 6]
            piece_emb[:, None, :, :].repeat(N, axis=1),       # [B, N, N, 6]
            same_color                                        # [B, N, N, 1]
        ], axis=-1)
        
        # Project to per-head biases [B, N, N, num_heads]
        bias = self.spatial_mlp(pairwise_feats)
        return jnp.transpose(bias, (0, 3, 1, 2))  # [B, num_heads, N, N]

class PhaseAwareModule(hk.Module):
    """Phase-aware positional bias scaling (Haiku version)"""
    def __init__(self, num_heads, phase_embed_dim=16, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.phase_embed_dim = phase_embed_dim

    def __call__(self, x, positional_bias):
        # 1. Game phase summary (attention pooling)
        attn_scores = hk.Linear(1)(x)  # [B, seq_len, 1]
        attn_weights = jax.nn.softmax(attn_scores, axis=1)
        phase_summary = jnp.sum(attn_weights * x, axis=1)  # [B, D]

        # 2. Phase encoding
        phase_emb = hk.Sequential([
            hk.Linear(self.phase_embed_dim),
            jax.nn.gelu
        ])(phase_summary)

        # 3. Scaling factors
        scales = jax.nn.sigmoid(hk.Linear(self.num_heads)(phase_emb))  # [B, H]
        
        return positional_bias * scales[:, :, None, None]  # [B, H, seq_len, seq_len]


class SmolgenModule(hk.Module):
  """Smolgen module for generating dynamic positional attention biases."""

  def __init__(
      self,
      num_heads: int,
      sequence_length: int = 64,
      compress_dim: int = 32,
      summary_dim: int = 256,
      name: str | None = None,
  ) -> None:
    """Initializes the Smolgen module.

    Args:
      num_heads: Number of attention heads.
      sequence_length: Length of input sequence (64 for chess).
      compress_dim: Dimension to compress each token to.
      summary_dim: Dimension of the position summary vector.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._sequence_length = sequence_length
    self._compress_dim = compress_dim
    self._summary_dim = summary_dim

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Generates supplemental attention logits from input representations.

    Args:
      inputs: Input token representations of shape [batch, seq_len, embed_dim].

    Returns:
      Supplemental attention logits of shape [batch, num_heads, seq_len, seq_len].
    """
    batch_size, sequence_length, embedding_dim = inputs.shape

    # Step 1: Compress position representation
    # Project each token to compress_dim
    compressed = hk.Linear(self._compress_dim, with_bias=False, name="compress")(inputs)
    # Shape: [batch, seq_len, compress_dim]

    # Flatten to single vector per position
    flattened = jnp.reshape(compressed, (batch_size, sequence_length * self._compress_dim))
    # Shape: [batch, seq_len * compress_dim]

    # Dense layer to extract position summary
    position_summary = hk.Linear(
        self._summary_dim, with_bias=True, name="position_dense"
    )(flattened)
    # Apply activation
    position_summary = jnn.silu(position_summary)
    # Shape: [batch, summary_dim]

    # Step 2: Generate head-specific attention logits
    # Get shared weight matrix (shared across all layers and heads)
    shared_projection = hk.get_parameter(
        "shared_projection",
        shape=(self._summary_dim, self._sequence_length * self._sequence_length),
        init=hk.initializers.RandomNormal(stddev=0.02),
    )

    supplemental_logits_all = []
    for head_idx in range(self._num_heads):
      # Head-specific transformation of position summary
      head_summary = hk.Linear(
          self._summary_dim, 
          with_bias=True, 
          name=f"head_{head_idx}_projection"
      )(position_summary)
      head_summary = jnn.silu(head_summary)
      # Shape: [batch, summary_dim]

      # Generate supplemental attention logits using shared projection
      head_logits = jnp.dot(head_summary, shared_projection)
      # Shape: [batch, seq_len * seq_len]

      # Reshape to attention matrix
      head_logits = jnp.reshape(
          head_logits, (batch_size, self._sequence_length, self._sequence_length)
      )
      # Shape: [batch, seq_len, seq_len]

      supplemental_logits_all.append(head_logits)

    # Stack all heads
    supplemental_logits = jnp.stack(supplemental_logits_all, axis=1)
    # Shape: [batch, num_heads, seq_len, seq_len]

    return supplemental_logits

class SmolgenModule2(hk.Module):
    def __init__(self,
                 sequence_length: int = 79,
                 num_heads: int = 8,
                 compress_dim: int = 64,
                 summary_dim: int = 128,
                 bias_rank: int = 32,  # for low-rank factorization
                 name: str = "SmolgenModule"):
        super().__init__(name=name)
        self.seq_len = sequence_length
        self.num_heads = num_heads
        self.compress_dim = compress_dim
        self.summary_dim = summary_dim
        self.bias_rank = bias_rank

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: [B, 79, D]
        B, S, D = inputs.shape
        special_tokens = inputs[:, 65:, :]  # [B, 15, D]

        # 1. Compress full sequence
        compressed = hk.Linear(self.compress_dim)(inputs)  # [B, 79, compress_dim]

        # 2. Reshape first 64 to 8x8 board and run CNN
        # CNN-based local board summary
        board = jnp.reshape(compressed[:, 1:65, :], [B, 8, 8, self.compress_dim])
        cnn_out = hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME')(board)
        cnn_out = jnn.silu(cnn_out)
        cnn_out = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME')(cnn_out)
        cnn_out = jnn.silu(cnn_out)
        cnn_flat = jnp.reshape(cnn_out, [B, -1])  # [B, F]

        # Global pooling (mean or attention pooling)
        compressed = jnp.reshape(compressed, [B, S* self.compress_dim])  # [B, S*C]
        global_tokens = hk.Linear(self.summary_dim)(compressed) 
        global_pool = jnn.silu(global_tokens)

        # 3. Position summary
        position_summary = jnp.concatenate([cnn_flat, global_pool], axis=-1)
        position_summary = hk.Linear(self.summary_dim)(position_summary)
        position_summary = jnn.silu(position_summary)

        # 4. Low-rank factorization to get per-head biases
        all_logits = []
        for h in range(self.num_heads):
            # Project to low-rank query/key factors
            q_proj = hk.Linear(self.seq_len * self.bias_rank, name=f'q_proj_{h}')(position_summary)
            k_proj = hk.Linear(self.seq_len * self.bias_rank, name=f'k_proj_{h}')(position_summary)

            q_factors = jnp.reshape(q_proj, [B, self.seq_len, self.bias_rank])
            k_factors = jnp.reshape(k_proj, [B, self.seq_len, self.bias_rank])

            # Compute low-rank bias: einsum('bsi,bti->bst')
            head_bias = jnp.einsum('bsi,bti->bst', q_factors, k_factors)
            all_logits.append(head_bias)

        # Stack heads: [B, H, S, S]
        return jnp.stack(all_logits, axis=1)

class SmolgenModule3(hk.Module):
  """Smolgen module for generating dynamic positional attention biases."""

  def __init__(
      self,
      num_heads: int,
      sequence_length: int = 64,
      summary_dim: int = 256,
      name: str | None = None,
  ) -> None:
    """Initializes the Smolgen module.

    Args:
      num_heads: Number of attention heads.
      sequence_length: Length of input sequence (64 for chess).
      summary_dim: Dimension of the position summary vector.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._sequence_length = sequence_length
    self._summary_dim = summary_dim

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Generates supplemental attention logits from input representations.

    Args:
      inputs: Input token representations of shape [batch, seq_len, embed_dim].

    Returns:
      Supplemental attention logits of shape [batch, num_heads, seq_len, seq_len].
    """
    batch_size, sequence_length, embedding_dim = inputs.shape
    
    # Step 1: Compress position representation using CNN
    # Extract only the chess board tokens (indices 1-64, corresponding to the 64 squares)
    board_tokens = inputs[:, 1:65, :]  # Shape: [batch, 64, embed_dim]
    actual_board_size = board_tokens.shape[1]  # Should be 64, but let's be safe
    
    # Reshape to 8x8 grid for CNN processing (64 chess board squares)
    # Ensure we have exactly 64 tokens for 8x8 grid
    if actual_board_size != 64:
        raise ValueError(f"Expected 64 board tokens, got {actual_board_size}")
    
    board_2d = jnp.reshape(
        board_tokens, 
        (batch_size, 8, 8, embedding_dim)
    )
    # Shape: [batch, 8, 8, embedding_dim]
    
    # Apply small CNN to extract position summary
    # First conv layer
    conv1 = hk.Conv2D(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=True,
        name="conv1"
    )(board_2d)
    conv1 = jnn.silu(conv1)
    # Shape: [batch, 8, 8, 64]
    
    # Second conv layer with stride 2 for downsampling
    conv2 = hk.Conv2D(
        output_channels=128,
        kernel_shape=3,
        stride=2,
        padding="SAME",
        with_bias=True,
        name="conv2"
    )(conv1)
    conv2 = jnn.silu(conv2)
    # Shape: [batch, 4, 4, 128]
    
    # Third conv layer with stride 2 for further downsampling
    conv3 = hk.Conv2D(
        output_channels=256,
        kernel_shape=3,
        stride=2,
        padding="SAME",
        with_bias=True,
        name="conv3"
    )(conv2)
    conv3 = jnn.silu(conv3)
    # Shape: [batch, 2, 2, 256]
    
    # Global average pooling to get final position summary
    position_summary = jnp.mean(conv3, axis=(1, 2))
    # Shape: [batch, 256]
    
    # Ensure the output dimension matches summary_dim
    if position_summary.shape[-1] != self._summary_dim:
      position_summary = hk.Linear(
          self._summary_dim, 
          with_bias=True, 
          name="summary_projection"
      )(position_summary)
    # Shape: [batch, summary_dim]

    # Step 2: Generate head-specific attention logits
    # Get shared weight matrix (shared across all layers and heads)
    # Only generate logits for board-to-board interactions
    board_logits_size = actual_board_size * actual_board_size
    shared_projection = hk.get_parameter(
        "shared_projection",
        shape=(self._summary_dim, board_logits_size),
        init=hk.initializers.RandomNormal(stddev=0.02),
    )

    supplemental_logits_all = []
    for head_idx in range(self._num_heads):
      # Head-specific transformation of position summary
      head_summary = hk.Linear(
          self._summary_dim, 
          with_bias=True, 
          name=f"head_{head_idx}_projection"
      )(position_summary)
      head_summary = jnn.silu(head_summary)
      # Shape: [batch, summary_dim]

      # Generate supplemental attention logits using shared projection
      head_logits_flat = jnp.dot(head_summary, shared_projection)
      # Shape: [batch, board_logits_size]

      # Reshape to board-to-board attention matrix
      head_logits_board = jnp.reshape(head_logits_flat, (batch_size, actual_board_size, actual_board_size))
      # Shape: [batch, actual_board_size, actual_board_size]
      
      # Pad to full sequence length with zeros for non-board tokens
      # Create full attention matrix initialized with zeros using actual sequence length
      head_logits_full = jnp.zeros((batch_size, sequence_length, sequence_length))
      
      # Fill in the board-to-board interactions (tokens 1 to 1+actual_board_size)
      end_idx = 1 + actual_board_size
      
      head_logits_full = head_logits_full.at[:, 1:end_idx, 1:end_idx].set(head_logits_board)
      # Shape: [batch, sequence_length, sequence_length]

      supplemental_logits_all.append(head_logits_full)

    # Stack all heads
    supplemental_logits = jnp.stack(supplemental_logits_all, axis=1)
    # Shape: [batch, num_heads, seq_len, seq_len]

    return supplemental_logits

class ChessGraphBuilder:
    """Builds chess graph from FEN representation."""
    
    @staticmethod
    def square_to_index(square: int) -> int:
        """Convert chess.Square to 0-63 index."""
        return square

    @staticmethod
    def index_to_coords(index: int) -> Tuple[int, int]:
        """Convert 0-63 index to (row, col) coordinates."""
        return index // 8, index % 8

    # Removed piece value and square value methods since we're using binary connectivity

    @staticmethod
    def build_adjacency_from_fen(fen: str) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Build adjacency matrices from FEN string using binary connectivity.
        
        Returns:
            attack_adj: [64, 64] attack adjacency matrix (1.0 if attack exists, 0.0 otherwise)
            defense_adj: [64, 64] defense adjacency matrix (1.0 if defense exists, 0.0 otherwise)
            movement_adj: [64, 64] movement adjacency matrix (1.0 if move possible, 0.0 otherwise)
        """
        board = chess.Board(fen)
        
        attack_adj = np.zeros((64, 64), dtype=np.float32)
        defense_adj = np.zeros((64, 64), dtype=np.float32)
        movement_adj = np.zeros((64, 64), dtype=np.float32)
        
        for square in range(64):
            piece = board.piece_at(square)
            if not piece:
                continue
                
            # Get all squares this piece attacks
            attacks = board.attacks(square)
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                
                if target_piece:
                    if target_piece.color != piece.color:
                        # Attack edge - binary connectivity
                        attack_adj[square, target_square] = 1.0
                    else:
                        # Defense edge - binary connectivity
                        defense_adj[square, target_square] = 1.0
                else:
                    # Movement edge - binary connectivity
                    movement_adj[square, target_square] = 1.0
        
        return jnp.array(attack_adj), jnp.array(defense_adj), jnp.array(movement_adj)


class ChessGraphNeuralNetwork(hk.Module):
    """Graph Neural Network for chess position reasoning."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        message_passing_steps: int = 2,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.message_passing_steps = message_passing_steps
    
    def __call__(
        self, 
        node_features: jax.Array,  # [batch, 64, embed_dim] - board squares only
        attack_adj: jax.Array,     # [batch, 64, 64]
        defense_adj: jax.Array,    # [batch, 64, 64] 
        movement_adj: jax.Array,   # [batch, 64, 64]
    ) -> jax.Array:
        """
        Apply GNN to chess board representation.
        
        Args:
            node_features: Node features for 64 squares [batch, 64, embed_dim]
            attack_adj: Attack adjacency matrix [batch, 64, 64]
            defense_adj: Defense adjacency matrix [batch, 64, 64]
            movement_adj: Movement adjacency matrix [batch, 64, 64]
            
        Returns:
            Updated node features [batch, 64, hidden_dim]
        """
        batch_size, num_nodes, input_dim = node_features.shape
        
        # Initial projection to hidden dimension
        h = hk.Linear(self.hidden_dim, name="input_projection")(node_features)
        
        # Message passing layers
        for layer_idx in range(self.num_layers):
            h = self._gnn_layer(
                h, attack_adj, defense_adj, movement_adj, 
                layer_idx=layer_idx
            )
        
        return h
    
    def _gnn_layer(
        self,
        node_features: jax.Array,  # [batch, 64, hidden_dim]
        attack_adj: jax.Array,     # [batch, 64, 64]
        defense_adj: jax.Array,    # [batch, 64, 64]
        movement_adj: jax.Array,   # [batch, 64, 64]
        layer_idx: int,
    ) -> jax.Array:
        """Single GNN layer with multi-relation message passing."""
        
        residual = node_features
        
        # Process each edge type separately
        attack_messages = self._compute_messages(
            node_features, attack_adj, f"attack_layer_{layer_idx}"
        )
        defense_messages = self._compute_messages(
            node_features, defense_adj, f"defense_layer_{layer_idx}"
        )
        movement_messages = self._compute_messages(
            node_features, movement_adj, f"movement_layer_{layer_idx}"
        )
        
        # Aggregate messages from all edge types
        aggregated_messages = attack_messages + defense_messages + movement_messages
        
        # Update node features
        updated_features = hk.Linear(
            self.hidden_dim, 
            name=f"update_layer_{layer_idx}"
        )(jnp.concatenate([node_features, aggregated_messages], axis=-1))
        
        updated_features = jnn.relu(updated_features)
        
        # Residual connection
        if updated_features.shape == residual.shape:
            updated_features = updated_features + residual
        
        # Layer normalization
        updated_features = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name=f"layernorm_{layer_idx}"
        )(updated_features)
        
        return updated_features
    
    def _compute_messages(
        self,
        node_features: jax.Array,  # [batch, 64, hidden_dim]
        adjacency: jax.Array,      # [batch, 64, 64]
        layer_name: str,
    ) -> jax.Array:
        """Compute messages for a specific edge type."""
        
        # Transform node features for message computation
        messages = hk.Linear(
            self.hidden_dim, 
            name=f"{layer_name}_message_transform"
        )(node_features)
        
        # Apply adjacency matrix to aggregate messages
        # adjacency[i,j] represents edge weight from node j to node i
        aggregated_messages = jnp.einsum('bij,bjd->bid', adjacency, messages)
        
        return aggregated_messages


class GNNSmolgenModule(hk.Module):
    """Smolgen module enhanced with GNN for chess-specific attention biases."""

    def __init__(
        self,
        num_heads: int,
        sequence_length: int = 79,  # Full sequence length including non-board tokens
        compress_dim: int = 32,
        summary_dim: int = 256,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._num_heads = num_heads
        self._sequence_length = sequence_length
        self._compress_dim = compress_dim
        self._summary_dim = summary_dim
        
        self._gnn = ChessGraphNeuralNetwork(
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            name="chess_gnn"
        )

    def __call__(
        self, 
        inputs: jax.Array,           # [batch, 79, embed_dim]
        fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ) -> jax.Array:
        """
        Generate attention biases using GNN on chess graph.
        
        Args:
            inputs: Full input sequence [batch, 79, embed_dim]
            fen_string: FEN string for current position
            
        Returns:
            Supplemental attention logits [batch, num_heads, 79, 79]
        """
        batch_size, sequence_length, embedding_dim = inputs.shape
        
        # Extract board squares (positions 1-64 in the sequence)
        board_features = inputs[:, 1:65, :]  # [batch, 64, embed_dim]
        
        # Build chess graph adjacency matrices
        attack_adj, defense_adj, movement_adj = ChessGraphBuilder.build_adjacency_from_fen(fen_string)
        
        # Expand adjacency matrices to batch dimension
        attack_adj = jnp.expand_dims(attack_adj, 0).repeat(batch_size, axis=0)
        defense_adj = jnp.expand_dims(defense_adj, 0).repeat(batch_size, axis=0)
        movement_adj = jnp.expand_dims(movement_adj, 0).repeat(batch_size, axis=0)
        
        # Apply GNN to board squares
        gnn_features = self._gnn(board_features, attack_adj, defense_adj, movement_adj)
        # gnn_features: [batch, 64, gnn_hidden_dim]
        
        # Process non-board tokens with standard layers
        non_board_features = inputs[:, [0] + list(range(65, 79)), :]  # [batch, 15, embed_dim]
        non_board_processed = hk.Linear(
            self._compress_dim, 
            name="non_board_compress"
        )(non_board_features)
        
        # Compress GNN features
        gnn_compressed = hk.Linear(
            self._compress_dim,
            name="gnn_compress"
        )(gnn_features)
        
        # Combine all features
        # Reconstruct full sequence: [special_token, 64_board_squares, 14_other_tokens]
        special_token_features = non_board_processed[:, :1, :]  # [batch, 1, compress_dim]
        other_token_features = non_board_processed[:, 1:, :]    # [batch, 14, compress_dim]
        
        combined_features = jnp.concatenate([
            special_token_features,    # position 0
            gnn_compressed,           # positions 1-64
            other_token_features      # positions 65-78
        ], axis=1)  # [batch, 79, compress_dim]
        
        # Flatten for position summary
        flattened = jnp.reshape(combined_features, (batch_size, -1))
        
        # Generate position summary
        position_summary = hk.Linear(self._summary_dim, name="position_dense")(flattened)
        position_summary = jnn.silu(position_summary)
        
        # Generate head-specific attention logits
        shared_projection = hk.get_parameter(
            "shared_projection",
            shape=(self._summary_dim, self._sequence_length * self._sequence_length),
            init=hk.initializers.RandomNormal(stddev=0.02),
        )

        supplemental_logits_all = []
        for head_idx in range(self._num_heads):
            head_summary = hk.Linear(
                self._summary_dim, 
                with_bias=True, 
                name=f"head_{head_idx}_projection"
            )(position_summary)
            head_summary = jnn.silu(head_summary)

            head_logits = jnp.dot(head_summary, shared_projection)
            head_logits = jnp.reshape(
                head_logits, (batch_size, self._sequence_length, self._sequence_length)
            )
            supplemental_logits_all.append(head_logits)

        supplemental_logits = jnp.stack(supplemental_logits_all, axis=1)
        return supplemental_logits

class ConvSmolgenModule(hk.Module):
  def __init__(
      self,
      num_heads: int,
      sequence_length: int = 64,
      compress_dim: int = 32,
      summary_dim: int = 256,
      conv_channels: tuple[int, ...] = (64, 128),
      name: str | None = None,
  ) -> None:
    """Initializes the Smolgen module.

    Args:
      num_heads: Number of attention heads.
      sequence_length: Length of input sequence (64 for chess).
      compress_dim: Dimension to compress each token to.
      summary_dim: Dimension of the position summary vector.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._sequence_length = sequence_length
    self._compress_dim = compress_dim
    self._summary_dim = summary_dim
    self._conv_channels = conv_channels

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """
    inputs: [batch, 79, embed_dim] where:
    - inputs[:, 0, :] = special token (ignore or use separately)
    - inputs[:, 1:65, :] = 64 board squares (8x8 grid)
    - inputs[:, 65:79, :] = castling, en passant, etc. (global features)
    """
    batch_size, sequence_length, embedding_dim = inputs.shape
    
    # Split the input into components
    board_tokens = inputs[:, 1:65, :]  # [batch, 64, embed_dim] - chess board
    global_tokens = inputs[:, 65:79, :] # [batch, 14, embed_dim] - game state
    
    # Process board spatially with CNN
    board_features = self._process_board_cnn(board_tokens)
    
    # Process global information with standard layers
    global_features = self._process_global_features(global_tokens)
    
    # Combine both feature types
    combined_features = jnp.concatenate([board_features, global_features], axis=-1)
    
    # Generate position summary
    position_summary = hk.Linear(self._summary_dim)(combined_features)
    position_summary = jnn.silu(position_summary)
    
    # Step 2: Generate head-specific attention logits
    # Get shared weight matrix (shared across all layers and heads)
    shared_projection = hk.get_parameter(
        "shared_projection",
        shape=(self._summary_dim, self._sequence_length * self._sequence_length),
        init=hk.initializers.RandomNormal(stddev=0.02),
    )

    supplemental_logits_all = []
    for head_idx in range(self._num_heads):
      # Head-specific transformation of position summary
      head_summary = hk.Linear(
          self._summary_dim, 
          with_bias=True, 
          name=f"head_{head_idx}_projection"
      )(position_summary)
      head_summary = jnn.silu(head_summary)
      # Shape: [batch, summary_dim]

      # Generate supplemental attention logits using shared projection
      head_logits = jnp.dot(head_summary, shared_projection)
      # Shape: [batch, seq_len * seq_len]

      # Reshape to attention matrix
      head_logits = jnp.reshape(
          head_logits, (batch_size, self._sequence_length, self._sequence_length)
      )
      # Shape: [batch, seq_len, seq_len]

      supplemental_logits_all.append(head_logits)

    # Stack all heads
    supplemental_logits = jnp.stack(supplemental_logits_all, axis=1)
    # Shape: [batch, num_heads, seq_len, seq_len]

    return supplemental_logits


  def _process_board_cnn(self, board_tokens: jax.Array) -> jax.Array:
    """Process 64 board squares with CNN."""
    batch_size = board_tokens.shape[0]
    
    # Reshape to 8x8 spatial format
    spatial_board = jnp.reshape(board_tokens, (batch_size, 8, 8, -1))
    
    # Apply convolutional layers
    h = spatial_board
    for i, channels in enumerate(self._conv_channels):
      h = hk.Conv2D(
          output_channels=channels,
          kernel_shape=3,
          padding='SAME',
          name=f"board_conv_{i}"
      )(h)
      h = jnn.relu(h)
    
    # Global average pooling
    board_summary = jnp.mean(h, axis=(1, 2))  # [batch, final_channels]
    return board_summary

  def _process_global_features(self, global_tokens: jax.Array) -> jax.Array:
    """Process castling, en passant, etc. with standard layers."""
    # Flatten global features
    flattened = jnp.reshape(global_tokens, (global_tokens.shape[0], -1))
    
    # Process with MLP
    h = hk.Linear(256, name="global_dense1")(flattened)
    h = jnn.relu(h)
    h = hk.Linear(128, name="global_dense2")(h)
    h = jnn.relu(h)
    
    return h  # [batch, 128]

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn

class BetterSmolgenModule(hk.Module):
    """Efficient version with factorized projections and hierarchical processing."""
    
    def __init__(
        self,
        num_hiddens_per_head: int,
        sequence_length: int,
        compress_dim: int = 32,
        summary_dim: int = 256,
        intermediate_dim: int = 256,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._d = num_hiddens_per_head
        self._T = sequence_length
        self._compress_dim = compress_dim
        self._summary_dim = summary_dim
        self._intermediate_dim = intermediate_dim

    def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        B, T, _ = inputs.shape

        # 1. Initial compression
        compressed = hk.Linear(self._compress_dim)(inputs)
        compressed = jnn.gelu(compressed)
        
        # 2. Position-aware processing
        # Split into two paths: position-independent and position-aware
        pos_independent = hk.Linear(self._compress_dim)(compressed.mean(axis=1, keepdims=True))
        pos_aware = hk.Linear(self._compress_dim)(compressed)
        
        # Combine with residual
        processed = pos_independent + pos_aware
        processed = jnn.silu(processed)
        
        # 3. Factorized Q/K/V generation
        def _make_bias_network(summary_type: str):
            """Creates a factorized projection network for one bias type."""
            # First project to intermediate dimension
            proj1 = hk.Linear(self._intermediate_dim, name=f"{summary_type}_proj1")
            # Then project to T*d (but factorized as T and d)
            proj2_t = hk.Linear(T, name=f"{summary_type}_proj_t")
            proj2_d = hk.Linear(self._d, name=f"{summary_type}_proj_d")
            
            def network(x):
                h = proj1(x)
                h = jnn.gelu(h)
                # Factorized projection: (B,T,C) -> (B,T,intermediate) -> (B,T,T) and (B,T,d)
                t_part = proj2_t(h)  # [B,T,T]
                d_part = proj2_d(h)  # [B,T,d]
                # Combine with outer product
                return jnp.einsum('bti,btj->btij', t_part, d_part)  # [B,T,T,d]
            
            return network
        
        # Create separate networks for each bias type
        q_net = _make_bias_network("q")
        k_net = _make_bias_network("k")
        v_net = _make_bias_network("v")
        
        # Generate each bias
        a_q = q_net(processed)
        a_k = k_net(processed)
        a_v = v_net(processed)
        
        # 4. Normalized output
        scale = 1.0 / jnp.sqrt(self._d)
        return (a_q * scale, a_k * scale, a_v * scale)


class BetterSmolgenModule2(hk.Module):
    """Efficient Smolgen for generating per-token-pair vectors with head sharing."""
    
    def __init__(
        self,
        num_heads: int,
        sequence_length: int = 64,
        compress_dim: int = 32,
        summary_dim: int = 256,
        head_dim: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._num_heads = num_heads
        self._sequence_length = sequence_length
        self._compress_dim = compress_dim
        self._summary_dim = summary_dim
        self._head_dim = head_dim

    def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size, sequence_length, embedding_dim = inputs.shape
        
        # Calculate head dimension if not provided
        if self._head_dim is None:
            self._head_dim = embedding_dim // self._num_heads
            if self._head_dim * self._num_heads != embedding_dim:
                raise ValueError(
                    f"embedding_dim {embedding_dim} must be divisible by num_heads {self._num_heads}"
                )

        # Step 1: Compress input tokens
        compressed = hk.Linear(self._compress_dim, with_bias=False, name="compress")(inputs)
        
        # Step 2: Create global sequence summary
        flattened = jnp.reshape(compressed, (batch_size, sequence_length * self._compress_dim))
        position_summary = hk.Linear(self._summary_dim, with_bias=True, name="position_dense")(flattened)
        position_summary = jnn.silu(position_summary)
        
        # Step 3: Create per-token-pair basis
        # [B, T, T, summary_dim]
        position_summary = position_summary[:, None, None, :]
        position_summary = jnp.tile(position_summary, (1, sequence_length, sequence_length, 1))
        
        # Add relative position information
        relative_bias = hk.get_parameter(
            "relative_bias",
            shape=(sequence_length, sequence_length, self._summary_dim),
            init=hk.initializers.RandomNormal(stddev=0.02),
        )
        pair_basis = position_summary + relative_bias[None, ...]
        pair_basis = jnn.silu(pair_basis)
        
        # Step 4: Generate shared basis vectors
        shared_basis = hk.Linear(
            self._summary_dim,  # Output dimension matches input for efficiency
            with_bias=True,
            name="shared_basis"
        )(pair_basis)
        shared_basis = jnn.silu(shared_basis)
        # Shape: [B, T, T, summary_dim]

        # Step 5: Head-specific projections
        # We'll create three separate projections for q, k, v
        def create_head_vectors(prefix: str):
            """Creates head-specific vectors from shared basis."""
            # Project to head_dim (shared across heads)
            projection = hk.Linear(
                self._num_heads * self._head_dim,
                with_bias=True,
                name=f"{prefix}_projection"
            )(shared_basis)
            
            # Reshape to [B, T, T, num_heads, head_dim]
            return jnp.reshape(
                projection,
                (batch_size, sequence_length, sequence_length, self._num_heads, self._head_dim)
            )
        
        a_q = create_head_vectors("query")
        a_k = create_head_vectors("key")
        a_v = create_head_vectors("value")
        
        # Convert to [B, T, T, d] where d = num_heads * head_dim
        a_q = jnp.reshape(a_q, (batch_size, sequence_length, sequence_length, -1))
        a_k = jnp.reshape(a_k, (batch_size, sequence_length, sequence_length, -1))
        a_v = jnp.reshape(a_v, (batch_size, sequence_length, sequence_length, -1))

        return a_q, a_k, a_v
    
import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
from typing import Tuple

class BetterSmolgenModule3(hk.Module):
  """
  Smolgen module based on user's proposal:
  Build rich token-level representations with global context, then derive shared biases.
  """
  def __init__(
      self,
      head_dim: int,
      sequence_length: int = 64,
      compress_dim: int = 32,        # Initial compression dim C_comp
      ffn_global_bottleneck_dim: int = 256, # Bottleneck for the global FFN
      s_tok_dim: int = 64,          # Dimension of processed_sequence_tokens
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._head_dim = head_dim
    self._sequence_length = sequence_length
    self._compress_dim = compress_dim
    self._ffn_global_bottleneck_dim = ffn_global_bottleneck_dim
    self._s_tok_dim = s_tok_dim

    self._weight_initializer = hk.initializers.RandomNormal(stddev=0.02)
    self._bias_initializer = hk.initializers.Constant(0.0)

  def _get_qk_interaction_params(self, name_suffix: str):
    # Projects from S_tok_dim to head_dim (applied token-wise)
    W_qf = hk.get_parameter(f"W_qf_{name_suffix}", shape=(self._s_tok_dim, self._head_dim), init=self._weight_initializer)
    b_qf = hk.get_parameter(f"b_qf_{name_suffix}", shape=(self._head_dim,), init=self._bias_initializer)
    W_kf = hk.get_parameter(f"W_kf_{name_suffix}", shape=(self._s_tok_dim, self._head_dim), init=self._weight_initializer)
    b_kf = hk.get_parameter(f"b_kf_{name_suffix}", shape=(self._head_dim,), init=self._bias_initializer)
    return W_qf, b_qf, W_kf, b_kf

  def __call__(self, inputs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    batch_size, _, _ = inputs.shape

    # 1. Initial Compression
    compressed_tokens = hk.Linear(self._compress_dim, with_bias=False, name="initial_compress")(inputs)
    # [B, T, C_comp]

    # 2. Flatten for Global FFN
    flattened_for_ffn = jnp.reshape(compressed_tokens, (batch_size, self._sequence_length * self._compress_dim))
    # [B, T*C_comp]

    # 3. Global FFN (with bottleneck)
    # Input dim: T*C_comp. Output dim: T*C_comp (to reshape back easily)
    ffn_input_dim = self._sequence_length * self._compress_dim
    hidden_global = hk.Linear(self._ffn_global_bottleneck_dim, name="ffn_global_1")(flattened_for_ffn)
    hidden_global = jnn.silu(hidden_global) # Activation
    ffn_output_flat = hk.Linear(ffn_input_dim, name="ffn_global_2")(hidden_global)
    # [B, T*C_comp]
    # Optional: Add residual connection
    # ffn_output_flat += flattened_for_ffn

    # 4. Reshape back to sequence
    globally_informed_sequence = jnp.reshape(ffn_output_flat, (batch_size, self._sequence_length, self._compress_dim))
    # [B, T, C_comp] (Assuming output_dim of FFN_global_2 was T*C_comp)

    # 5. Token-wise MLP to get S_tok_dim (example: one layer)
    # This layer processes each token's C_comp features to S_tok_dim features
    # Can be made deeper with more layers, activations, residuals.
    processed_sequence_tokens = hk.Linear(self._s_tok_dim, name="token_mlp_to_stok")(globally_informed_sequence)
    processed_sequence_tokens = jnn.silu(processed_sequence_tokens) # Activation
    # [B, T, S_tok_dim]

    # --- Generate a_q, a_k, a_v components ---
    params_q = self._get_qk_interaction_params("q")
    params_k = self._get_qk_interaction_params("k")
    params_v = self._get_qk_interaction_params("v")

    generated_components = []
    for W_qf, b_qf, W_kf, b_kf in [params_q, params_k, params_v]:
        # 6. Factor Generation (token-wise projection)
        q_factors = jnp.dot(processed_sequence_tokens, W_qf) + b_qf
        # q_factors: [B, T, head_dim]
        k_factors = jnp.dot(processed_sequence_tokens, W_kf) + b_kf
        # k_factors: [B, T, head_dim]

        # 7. Interaction
        component = q_factors[:, :, None, :] * k_factors[:, None, :, :]
        # component: [B, T, T, head_dim]
        generated_components.append(component)

    a_q, a_k, a_v = generated_components[0], generated_components[1], generated_components[2]
    return a_q, a_k, a_v


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention with optional Smolgen enhancement."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
      apply_qk_layernorm: bool = False,
      use_smolgen: bool = False,
      smolgen_compress_dim: int = 32,
      smolgen_summary_dim: int = 256,
      gnn_hidden_dim: int = 128,
      gnn_num_layers: int = 3,

  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
      apply_qk_layernorm: Applies layernorm to query and key matrices.
      use_smolgen: Whether to use Smolgen for dynamic attention biases.
      smolgen_compress_dim: Compression dimension for Smolgen.
      smolgen_summary_dim: Summary dimension for Smolgen.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._apply_qk_layernorm = apply_qk_layernorm
    self._use_smolgen = use_smolgen
    self.smolgen_mode = 8
    
    if self._use_smolgen:
      if self.smolgen_mode == 0:
        pass
        # relative position bias to add
      elif self.smolgen_mode in [1, 5]:
        # self._smolgen = SmolgenModule(
        #     num_heads=num_heads,
        #     sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
        #     compress_dim=smolgen_compress_dim,
        #     summary_dim=smolgen_summary_dim,
        #     name="smolgen",
        # )
        self._smolgen = ConvSmolgenModule(
            num_heads=num_heads,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            compress_dim=smolgen_compress_dim,
            summary_dim=smolgen_summary_dim,
            name="smolgen",
        )
      elif self.smolgen_mode == 3:
        self._smolgen = SmolgenModule(
            num_heads=num_heads,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            compress_dim=smolgen_compress_dim,
            summary_dim=smolgen_summary_dim,
            name="smolgen",
        )
        # self._prg = PRGModule(
        #     num_heads=num_heads,
        #     compress_dim=smolgen_compress_dim//2,
        #     prg_hidden_dim=smolgen_compress_dim,
        #     name="prg",
        # )
        self._prg = PRGModule2(
            num_heads=num_heads,
            name="prg",
        )
        self._phase_aware = PhaseAwareModule(
            num_heads=num_heads,
            phase_embed_dim=smolgen_compress_dim//2,  # Half the summary dim for phase encoding
            name="phase_aware",
        )
      elif self.smolgen_mode == 2:
        self._smolgen = BetterSmolgenModule(
            num_hiddens_per_head=32,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            compress_dim=smolgen_compress_dim,
            summary_dim=smolgen_summary_dim,
            name="smolgen",
        )
      elif self.smolgen_mode == 4:
         self._smolgen = BetterSmolgenModule3(
            head_dim=num_hiddens_per_head,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            compress_dim=smolgen_compress_dim,
            ffn_global_bottleneck_dim=smolgen_summary_dim,
            s_tok_dim=smolgen_compress_dim,
            name="smolgen",
         )
      elif self.smolgen_mode == 6:
         self._smolgen = SmolgenModule2(
            num_heads=num_heads,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            compress_dim=smolgen_compress_dim,
            summary_dim=smolgen_summary_dim,
            name="smolgen",
         )
      elif self.smolgen_mode == 7:
         self._smolgen = GNNSmolgenModule(
                num_heads=num_heads,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_num_layers=gnn_num_layers,
                compress_dim=smolgen_compress_dim,
                summary_dim=smolgen_summary_dim,
                name="gnn_smolgen",
            )
      elif self.smolgen_mode == 8:
         self._smolgen = SmolgenModule3(
            num_heads=num_heads,
            sequence_length=77 + 2,  # 64 squares + 13 global features + 2 special tokens
            summary_dim=smolgen_summary_dim,
            name="smolgen",
        )

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
      fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  ) -> jax.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)

    if self._apply_qk_layernorm:
      q = layer_norm(q)
      k = layer_norm(k)

    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    if self._use_smolgen and self.smolgen_mode in [0, 5]:
      # Relative position bias implementation
      a_q = hk.get_parameter(
          'relative_position_q',
          shape=(sequence_length, sequence_length, self._num_hiddens_per_head),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      a_k = hk.get_parameter(
          'relative_position_k',
          shape=(sequence_length, sequence_length, self._num_hiddens_per_head),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      a_v = hk.get_parameter(
          'relative_position_v',
          shape=(sequence_length, sequence_length, self._num_hiddens_per_head),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      
      # Basic attention (q*k)
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      
      # # Relative position terms
      # # q*a_k: [b,t,h,d] * [t,T,d] -> sum over d -> [b,h,t,T]
      # q_a_k = jnp.einsum('bthd,tTd->bhtT', q, a_k)
      
      # # a_q*k: [t,T,d] * [b,T,h,d] -> need to transpose a_q
      # a_q_transposed = jnp.transpose(a_q, axes=(1, 0, 2))  # [T,t,d]
      # a_q_k = jnp.einsum('Ttd,bThd->bhtT', a_q_transposed, k)
      
      # # a_q*a_k: sum over last dim after element-wise multiply
      # a_q_a_k = jnp.einsum('tTd,tTd->tT', a_q, a_k)  # [t,T]
      # a_q_a_k = a_q_a_k[None, None, :, :]  # [1,1,t,T] for broadcasting
      
      # # Combine all terms
      # attention += q_a_k + a_q_k + a_q_a_k
      attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    elif self._use_smolgen and self.smolgen_mode in [2, 4]:
      a_q, a_k, a_v = self._smolgen(inputs_q)
      # Basic attention (q*k)
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      
      # Relative position terms
      # q*a_k: [b,t,h,d] * [b, t,T,d] -> sum over d -> [b,h,t,T]
      q_a_k = jnp.einsum('bthd,btTd->bhtT', q, a_k)
      
      # a_q*k: [t,T,d] * [b,T,h,d] -> need to transpose a_q
      a_q_transposed = jnp.transpose(a_q, axes=(0, 2, 1, 3))  # [T,t,d]
      a_q_k = jnp.einsum('bTtd,bThd->bhtT', a_q_transposed, k)
      
      # a_q*a_k: sum over last dim after element-wise multiply
      a_q_a_k = jnp.einsum('btTd,btTd->btT', a_q, a_k)  # [t,T]
      a_q_a_k = a_q_a_k[:, None, :, :]  # [B,1,t,T] for broadcasting
      
      # Combine all terms
      attention += q_a_k + a_q_k + a_q_a_k
      attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)
    else:
      # Standard dot-product attention logits
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    if self._use_smolgen:
      # Generate dynamic positional attention biases using Smolgen
      if self.smolgen_mode in [1, 5, 6, 8]:
        supplemental_logits = self._smolgen(inputs_q)
        attention += supplemental_logits
      elif self.smolgen_mode == 7:
        supplemental_logits = self._smolgen(inputs_q, fen_string)
        attention += supplemental_logits
      elif self.smolgen_mode == 3:
        supplemental_logits = self._smolgen(inputs_q)
        prg_bias = self._prg(inputs_q)
        phase_bias = self._phase_aware(inputs_q, supplemental_logits)
        attention += phase_bias + prg_bias
    # else:
    #   # Original static positional bias
    #   position_bias = hk.get_parameter(
    #       'position_bias',
    #       shape=(self._num_heads, 77 + 2, 77 + 2),
    #       init=hk.initializers.RandomNormal(stddev=0.02),
    #   )
    #   attention += position_bias[None, :, :, :]

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    if self._use_smolgen and self.smolgen_mode in [0, 5]:
      output += jnp.einsum('bhtT,tTd->bthd', normalized_attention, a_v)
    elif self._use_smolgen and self.smolgen_mode in [2, 4]:
      output += jnp.einsum('bhtT,btTd->bthd', normalized_attention, a_v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


class GroupedQueryAttention(hk.Module):
  """Grouped Query Attention (GQA) without output tiling."""

  def __init__(
      self,
      num_heads: int,  # Number of KV heads
      num_query_groups: int,  # Number of Q groups
      num_hiddens_per_head: int,
      name: str | None = None,
      apply_qk_layernorm: bool = False,
  ) -> None:
    super().__init__(name=name)
    assert num_heads >= 1, "There must be at least one KV head."
    assert num_query_groups >= 1, "There must be at least one query group."
    assert num_heads % num_query_groups == 0 or num_query_groups % num_heads == 0, (
      "num_heads and num_query_groups must divide evenly for grouping or repeating."
    )

    self._num_heads = num_heads
    self._num_query_groups = num_query_groups
    self._num_hiddens_per_head = num_hiddens_per_head
    self._apply_qk_layernorm = apply_qk_layernorm

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
      fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  ) -> jax.Array:
    batch_size, seq_len, embedding_size = inputs_q.shape

    q = hk.Linear(self._num_query_groups * self._num_hiddens_per_head, with_bias=False)(inputs_q)
    k = hk.Linear(self._num_heads * self._num_hiddens_per_head, with_bias=False)(inputs_kv)
    v = hk.Linear(self._num_heads * self._num_hiddens_per_head, with_bias=False)(inputs_kv)

    if self._apply_qk_layernorm:
      q = layer_norm(q)
      k = layer_norm(k)

    q = q.reshape(batch_size, seq_len, self._num_query_groups, self._num_hiddens_per_head)
    k = k.reshape(batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    v = v.reshape(batch_size, -1, self._num_heads, self._num_hiddens_per_head)

    if self._num_heads < self._num_query_groups:
      # Repeat KV heads across Q groups
      repeat_factor = self._num_query_groups // self._num_heads
      k = jnp.repeat(k, repeat_factor, axis=2)
      v = jnp.repeat(v, repeat_factor, axis=2)
      k = k[:, :, :, None, :]
      v = v[:, :, :, None, :]
      q = q[:, :, :, None, :]
    elif self._num_heads > self._num_query_groups:
      # Split KV heads across Q groups
      kv_per_group = self._num_heads // self._num_query_groups
      k = k.reshape(batch_size, -1, self._num_query_groups, kv_per_group, self._num_hiddens_per_head)
      v = v.reshape(batch_size, -1, self._num_query_groups, kv_per_group, self._num_hiddens_per_head)
      q = q[:, :, :, None, :]
    else:
      # Equal number of heads and groups
      k = k[:, :, :, None, :]
      v = v[:, :, :, None, :]
      q = q[:, :, :, None, :]

    attn_scores = jnp.einsum("btghd,bTghd->btghT", q, k)
    attn_scores *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    if mask is not None:
      attn_scores = jnp.where(mask[:, None, None, None, :], attn_scores, jnp.finfo(jnp.float32).min)

    attn_weights = jnn.softmax(attn_scores, axis=-1)
    output = jnp.einsum("btghT,bTghd->btghd", attn_weights, v)
    output = output.reshape(batch_size, seq_len, self._num_query_groups * self._num_hiddens_per_head)

    return hk.Linear(embedding_size, with_bias=False)(output)

def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = np.arange(start=0, stop=sequence_length)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]


def embed_sequences(
    sequences: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns embeddings for sequences of tokens."""
  embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
  embeddings_layer = hk.Embed(
      vocab_size=config.vocab_size,
      embed_dim=config.embedding_dim,
      lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
      w_init=embs_init,
  )
  embeddings = embeddings_layer(sequences)
  embeddings *= jnp.sqrt(config.embedding_dim)

  _, sequence_length, embedding_size = embeddings.shape
  match config.pos_encodings:
    case PositionalEncodings.SINUSOID:
      pos_encodings = sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
      )
    case PositionalEncodings.LEARNED:
      assert sequence_length <= config.max_sequence_length
      positions = jnp.arange(sequence_length)
      pos_encodings = hk.Embed(
          vocab_size=config.max_sequence_length,
          embed_dim=embedding_size,
      )(positions)
  return embeddings + pos_encodings


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def _mlp_block(inputs: jax.Array, config: TransformerConfig) -> jax.Array:
  """Gated MLP block for the Transformer."""
  ffn_dim = int(config.embedding_dim * config.widening_factor)
  split_1 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  split_2 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  gate_output = jnn.gelu(split_1) * split_2
  return hk.Linear(config.embedding_dim, with_bias=False)(gate_output)


def _moe_block(inputs: jax.Array, config: TransformerConfig) -> jax.Array:
    """Corrected MoE implementation with proper RNG handling."""
    num_experts = 4
    k = 1
    batch_size, seq_len, emb_dim = inputs.shape
    inputs_flat = jnp.reshape(inputs, (-1, emb_dim))  # [batch*seq, emb]

    # Router network
    router_logits = hk.Linear(num_experts, with_bias=False, name="router")(inputs_flat)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    topk_weights, topk_indices = jax.lax.top_k(router_probs, k)
    topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)

    # Expert definition with explicit parameter scoping
    def expert_fn(expert_input: jax.Array) -> jax.Array:
        ffn_dim = int(config.embedding_dim * config.widening_factor/k)
        with hk.experimental.name_scope("expert"):  # Critical for param separation
            split_1 = hk.Linear(ffn_dim, with_bias=False, name="linear_1")(expert_input)
            split_2 = hk.Linear(ffn_dim, with_bias=False, name="linear_2")(expert_input)
            gate = jax.nn.silu(split_1) * split_2
            return hk.Linear(emb_dim, with_bias=False, name="linear_out")(gate)

    # Fixed vmap configuration
    experts = hk.vmap(
        expert_fn,
        in_axes=0,
        out_axes=0,
        axis_size=num_experts,
        split_rng=False  # Disable RNG split during init
    )

    # Process through experts
    expanded_inputs = jnp.broadcast_to(inputs_flat, (num_experts,) + inputs_flat.shape)
    expert_outputs = experts(expanded_inputs)  # [num_experts, batch*seq, emb]

    # Create routing mask
    token_indices = jnp.arange(inputs_flat.shape[0])
    expert_mask = jnp.zeros((inputs_flat.shape[0], num_experts), dtype=topk_weights.dtype)
    expert_mask = expert_mask.at[token_indices[:, None], topk_indices].set(topk_weights)

    # Combine outputs
    combined_flat = jnp.einsum('nte,tn->te', expert_outputs, expert_mask)
    return jnp.reshape(combined_flat, (batch_size, seq_len, emb_dim))


def _attention_block(inputs: jax.Array, config: TransformerConfig, fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") -> jax.Array:
  """Attention block for the Transformer."""
  batch_size, sequence_length = inputs.shape[:2]
  if config.use_causal_mask:
    causal_mask = np.tril(
        np.ones((batch_size, 1, sequence_length, sequence_length))
    )
  else:
    causal_mask = None
    block = MultiHeadDotProductAttention(
      num_heads=config.num_heads,
      num_hiddens_per_head=config.embedding_dim // config.num_heads,
      apply_qk_layernorm=config.apply_qk_layernorm,
      use_smolgen=config.use_smolgen,
      smolgen_compress_dim=config.smolgen_compress_dim,
      smolgen_summary_dim=config.smolgen_summary_dim,
    )
  #   block = GroupedQueryAttention(
  #       num_heads=config.num_heads // 2,
  #       num_query_groups=config.num_heads,
  #       num_hiddens_per_head=config.embedding_dim // config.num_heads,
  #       apply_qk_layernorm=True,
  #   )
  return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask, fen_string=fen_string)


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
    fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Follows the LLaMa architecture:
  https://github.com/facebookresearch/llama/blob/main/llama/model.py
  Main changes to the original Transformer decoder:
  - Using gating in the MLP block, with SwiGLU activation function.
  - Using normalization before the attention and MLP blocks.
  - Optional Smolgen for dynamic positional attention biases.

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  # Right shift the targets to get the inputs (the first token is now a 0).
  inputs = shift_right(targets)

  # Embeds the inputs and adds positional encodings.
  embeddings = embed_sequences(inputs, config)

  h = embeddings
  for _ in range(config.num_layers):
    attention_input = layer_norm(h)
    attention = _attention_block(attention_input, config, fen_string=fen_string)
    h += attention

    mlp_input = layer_norm(h)
    mlp_output = _mlp_block(mlp_input, config)
    h += mlp_output

  if config.apply_post_ln:
    h = layer_norm(h)
  logits = hk.Linear(config.output_size)(h)
  return jnn.log_softmax(logits, axis=-1)


def build_transformer_predictor(
    config: TransformerConfig,
) -> constants.Predictor:
  """Returns a transformer predictor."""
  model = hk.transform(functools.partial(transformer_decoder, config=config))
  return constants.Predictor(initial_params=model.init, predict=model.apply)
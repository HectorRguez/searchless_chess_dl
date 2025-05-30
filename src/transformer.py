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
import jax.numpy as jnp
import numpy as np

from searchless_chess.src import constants

import os
import os

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
  use_smolgen: bool = False
  # Compression dimension for Smolgen position summary.
  smolgen_compress_dim: int = 32
  # Position summary dimension for Smolgen.
  smolgen_summary_dim: int = 256

  def __post_init__(self):
    if self.output_size is None:
      self.output_size = self.vocab_size

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

class ConvSmolgenModule(hk.Module):
  """CNN-enhanced Smolgen module for chess with spatial board processing."""

  def __init__(
      self,
      num_heads: int,
      sequence_length: int = 79,  # Updated for chess: 1 + 64 + 14
      summary_dim: int = 128,     # Reduced from 256
      conv_channels: list[int] = None,
      name: str | None = None,
  ) -> None:
    """Initializes the CNN Smolgen module.

    Args:
      num_heads: Number of attention heads.
      sequence_length: Length of input sequence (79 for chess).
      summary_dim: Dimension of the position summary vector.
      conv_channels: List of channel dimensions for CNN layers.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._sequence_length = sequence_length
    self._summary_dim = summary_dim
    self._conv_channels = conv_channels or [16, 32]  # Much smaller: reduced from [64, 128, 256]
  
  def __call__(self, inputs: jax.Array) -> jax.Array:
    """
    Generates supplemental attention logits from input representations.
    
    Args:
      inputs: [batch, 79, embed_dim] where:
      - inputs[:, 0, :] = special token (ignore or use separately)
      - inputs[:, 1:65, :] = 64 board squares (8x8 grid)
      - inputs[:, 65:79, :] = castling, en passant, etc. (global features)
      
    Returns:
      Supplemental attention logits of shape [batch, num_heads, seq_len, seq_len].
    """
    batch_size, sequence_length, embedding_dim = inputs.shape
    
    # Split the input into components
    special_token = inputs[:, 0:1, :]       # [batch, 1, embed_dim] - special token
    board_tokens = inputs[:, 1:65, :]       # [batch, 64, embed_dim] - chess board
    global_tokens = inputs[:, 65:79, :]     # [batch, 14, embed_dim] - game state
    
    # Process board spatially with CNN
    board_features = self._process_board_cnn(board_tokens)
    
    # Process global information with standard layers
    global_features = self._process_global_features(global_tokens)
    
    # Process special token
    special_features = self._process_special_token(special_token)
    
    # Combine all feature types
    combined_features = jnp.concatenate([
        special_features, 
        board_features, 
        global_features
    ], axis=-1)
    
    # Generate position summary
    position_summary = hk.Linear(self._summary_dim, name="position_summary")(combined_features)
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
    
    # Process with smaller MLP
    h = hk.Linear(64, name="global_dense1")(flattened)  # Reduced from 256
    h = jnn.relu(h)
    h = hk.Linear(32, name="global_dense2")(h)          # Reduced from 128
    h = jnn.relu(h)
    
    return h  # [batch, 32]

  def _process_special_token(self, special_token: jax.Array) -> jax.Array:
    """Process the special token."""
    # Flatten and process
    flattened = jnp.reshape(special_token, (special_token.shape[0], -1))
    
    h = hk.Linear(16, name="special_dense")(flattened)  # Reduced from 64
    h = jnn.relu(h)
    
    return h  # [batch, 16]

class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention with optional CNN Smolgen enhancement."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
      apply_qk_layernorm: bool = False,
      use_smolgen: bool = False,
      use_cnn_smolgen: bool = False,
      smolgen_compress_dim: int = 32,
      smolgen_summary_dim: int = 128,           # Reduced default from 256
      smolgen_conv_channels: list[int] = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
      apply_qk_layernorm: Applies layernorm to query and key matrices.
      use_smolgen: Whether to use basic Smolgen for dynamic attention biases.
      use_cnn_smolgen: Whether to use CNN Smolgen (overrides use_smolgen if True).
      smolgen_compress_dim: Compression dimension for basic Smolgen.
      smolgen_summary_dim: Summary dimension for Smolgen.
      smolgen_conv_channels: Channel dimensions for CNN Smolgen.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._apply_qk_layernorm = apply_qk_layernorm
    self._use_smolgen = use_smolgen
    self._use_cnn_smolgen = use_cnn_smolgen
    
    # Priority: CNN Smolgen > Basic Smolgen > Static bias
    if self._use_cnn_smolgen:
      self._smolgen = ConvSmolgenModule(
          num_heads=num_heads,
          sequence_length=79,  # Chess sequence length
          summary_dim=smolgen_summary_dim,
          conv_channels=smolgen_conv_channels,
          name="cnn_smolgen",
      )
    elif self._use_smolgen:
      self._smolgen = SmolgenModule(
          num_heads=num_heads,
          compress_dim=smolgen_compress_dim,
          summary_dim=smolgen_summary_dim,
          name="smolgen",
      )

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
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

    # Standard dot-product attention logits
    attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    # Add positional biases based on configuration
    if self._use_cnn_smolgen or self._use_smolgen:
      # Generate dynamic positional attention biases using Smolgen
      supplemental_logits = self._smolgen(inputs_q)
      attention += supplemental_logits
    else:
      # Original static positional bias
      position_bias = hk.get_parameter(
          'position_bias',
          shape=(self._num_heads, sequence_length, sequence_length),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      attention += position_bias[None, :, :, :] 

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
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
  ffn_dim = config.embedding_dim * config.widening_factor
  split_1 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  split_2 = hk.Linear(ffn_dim, with_bias=False)(inputs)
  gate_output = jnn.silu(split_1) * split_2
  return hk.Linear(config.embedding_dim, with_bias=False)(gate_output)


def _attention_block(inputs: jax.Array, config: TransformerConfig, layer_idx: int = 0, log_file=None) -> jax.Array:
  """Attention block for the Transformer with e4 attention printing."""
def _attention_block(inputs: jax.Array, config: TransformerConfig, layer_idx: int = 0, log_file=None) -> jax.Array:
  """Attention block for the Transformer with e4 attention printing."""
  batch_size, sequence_length = inputs.shape[:2]
  if config.use_causal_mask:
    causal_mask = np.tril(
        np.ones((batch_size, 1, sequence_length, sequence_length))
    )
  else:
    causal_mask = None
  
  # Create a custom attention module that prints e4 attention
  class E4AttentionPrinter(hk.Module):
    def __init__(self, num_heads, num_hiddens_per_head, apply_qk_layernorm, layer_idx, log_file=None):
        super().__init__()
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self._apply_qk_layernorm = apply_qk_layernorm
        self._layer_idx = layer_idx
        self._log_file = log_file

    def __call__(self, inputs_q, inputs_kv, mask=None):
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

      # Compute attention logits
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

      # Add positional bias
      position_bias = hk.get_parameter(
          'position_bias',
          shape=(self._num_heads, 77 + 2, 77 + 2),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      attention += position_bias[None, :, :, :] 

      if mask is not None:
        attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

      # Apply softmax to get attention weights
      normalized_attention = jnn.softmax(attention)

      # PRINT E4 ATTENTION VALUES HERE
      # E4 square is at position 36 in the sequence (1 + 35, where 35 is 4*8+3)
      # But let's be more flexible and find the right position
      self._print_e4_attention(normalized_attention, sequence_length)

      # Continue with normal attention computation
      output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
      output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
      return hk.Linear(embedding_size, with_bias=False)(output)

    def _print_e4_attention(self, attention_weights, sequence_length):
      """Write attention values between e4 and the 64 board squares to file."""
      # E4 is at chess position (4,3) -> index 4*8+3 = 35 in 8x8 board
      # In sequence: [special_token] + [64 board squares] + [global features]
      # So e4 is at position 1+35 = 36 in the sequence
      e4_position = 36
      
      if e4_position >= sequence_length:
        error_msg = f"Warning: e4 position {e4_position} >= sequence length {sequence_length}"
        print(error_msg)
        if self._log_file:
          self._log_file.write(error_msg + "\n")
        return
      
      batch_size, num_heads, seq_len, _ = attention_weights.shape
      
      for head_idx in range(num_heads):
        # Get attention FROM e4 TO the 64 board squares (positions 1-64)
        e4_to_board = attention_weights[0, head_idx, e4_position, 1:65]
        
        # Get attention FROM the 64 board squares TO e4 (positions 1-64)
        board_to_e4 = attention_weights[0, head_idx, 1:65, e4_position]
        
        # Convert to regular Python lists for easy copying
        e4_to_board_list = e4_to_board.tolist()
        board_to_e4_list = board_to_e4.tolist()
        
        # Format the output lines
        e4_to_line = f"L{self._layer_idx}H{head_idx}_E4_TO_BOARD: {e4_to_board_list}"
        board_to_line = f"L{self._layer_idx}H{head_idx}_BOARD_TO_E4: {board_to_e4_list}"
        
        # Append to log.txt file
        with open("log.txt", "a") as log_file:
          log_file.write(e4_to_line + "\n")
          log_file.write(board_to_line + "\n")
          log_file.flush()  # Ensure immediate write
          
        # Write to file if available
        if self._log_file:
          self._log_file.write(e4_to_line + "\n")
          self._log_file.write(board_to_line + "\n")
          self._log_file.flush()  # Ensure immediate write

  block = E4AttentionPrinter(
  
  # Create a custom attention module that prints e4 attention
  class E4AttentionPrinter(hk.Module):
    def __init__(self, num_heads, num_hiddens_per_head, apply_qk_layernorm, layer_idx, log_file=None):
        super().__init__()
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self._apply_qk_layernorm = apply_qk_layernorm
        self._layer_idx = layer_idx
        self._log_file = log_file

    def __call__(self, inputs_q, inputs_kv, mask=None):
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

      # Compute attention logits
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
      attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

      # Add positional bias
      position_bias = hk.get_parameter(
          'position_bias',
          shape=(self._num_heads, 77 + 2, 77 + 2),
          init=hk.initializers.RandomNormal(stddev=0.02),
      )
      attention += position_bias[None, :, :, :] 

      if mask is not None:
        attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

      # Apply softmax to get attention weights
      normalized_attention = jnn.softmax(attention)

      # PRINT E4 ATTENTION VALUES HERE
      # E4 square is at position 36 in the sequence (1 + 35, where 35 is 4*8+3)
      # But let's be more flexible and find the right position
      self._print_e4_attention(normalized_attention, sequence_length)

      # Continue with normal attention computation
      output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
      output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
      return hk.Linear(embedding_size, with_bias=False)(output)

    def _print_e4_attention(self, attention_weights, sequence_length):
      """Write attention values between e4 and the 64 board squares to file."""
      # E4 is at chess position (4,3) -> index 4*8+3 = 35 in 8x8 board
      # In sequence: [special_token] + [64 board squares] + [global features]
      # So e4 is at position 1+35 = 36 in the sequence
      e4_position = 36
      
      if e4_position >= sequence_length:
        error_msg = f"Warning: e4 position {e4_position} >= sequence length {sequence_length}"
        print(error_msg)
        if self._log_file:
          self._log_file.write(error_msg + "\n")
        return
      
      batch_size, num_heads, seq_len, _ = attention_weights.shape
      
      for head_idx in range(num_heads):
        # Get attention FROM e4 TO the 64 board squares (positions 1-64)
        e4_to_board = attention_weights[0, head_idx, e4_position, 1:65]
        
        # Get attention FROM the 64 board squares TO e4 (positions 1-64)
        board_to_e4 = attention_weights[0, head_idx, 1:65, e4_position]
        
        # Convert to regular Python lists for easy copying
        e4_to_board_list = e4_to_board.tolist()
        board_to_e4_list = board_to_e4.tolist()
        
        # Format the output lines
        e4_to_line = f"L{self._layer_idx}H{head_idx}_E4_TO_BOARD: {e4_to_board_list}"
        board_to_line = f"L{self._layer_idx}H{head_idx}_BOARD_TO_E4: {board_to_e4_list}"
        
        # Append to log.txt file
        with open("log.txt", "a") as log_file:
          log_file.write(e4_to_line + "\n")
          log_file.write(board_to_line + "\n")
          log_file.flush()  # Ensure immediate write
          
        # Write to file if available
        if self._log_file:
          self._log_file.write(e4_to_line + "\n")
          self._log_file.write(board_to_line + "\n")
          self._log_file.flush()  # Ensure immediate write

  block = E4AttentionPrinter(
      num_heads=config.num_heads,
      num_hiddens_per_head=config.embedding_dim // config.num_heads,
      apply_qk_layernorm=config.apply_qk_layernorm,
      use_smolgen=config.use_smolgen,
      smolgen_compress_dim=config.smolgen_compress_dim,
      smolgen_summary_dim=config.smolgen_summary_dim,
  )
  return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask)

def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
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
  for layer_idx in range(config.num_layers):
  for layer_idx in range(config.num_layers):
    attention_input = layer_norm(h)
    attention = _attention_block(attention_input, config, layer_idx=layer_idx)  # Pass layer index
    attention = _attention_block(attention_input, config, layer_idx=layer_idx)  # Pass layer index
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
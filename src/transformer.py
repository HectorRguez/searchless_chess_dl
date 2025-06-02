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

  def __post_init__(self):
    if self.output_size is None:
      self.output_size = self.vocab_size

class SmolgenModule(hk.Module):
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
    
    if self._use_smolgen:
      self._smolgen = SmolgenModule(
          num_heads=num_heads,
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

    if self._use_smolgen:
      # Generate dynamic positional attention biases using Smolgen
      supplemental_logits = self._smolgen(inputs_q)
      attention += supplemental_logits
    else:
      # Original static positional bias
      position_bias = hk.get_parameter(
          'position_bias',
          shape=(self._num_heads, 77 + 2, 77 + 2),
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


def _attention_block(inputs: jax.Array, config: TransformerConfig) -> jax.Array:
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
  for _ in range(config.num_layers):
    attention_input = layer_norm(h)
    attention = _attention_block(attention_input, config)
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
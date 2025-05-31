import dataclasses
import enum
import functools
from typing import Tuple, List, Dict, Optional

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import chess


class PositionalEncodings(enum.Enum):
    SINUSOID = enum.auto()
    LEARNED = enum.auto()


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""
    seed: int = 1
    vocab_size: int
    output_size: int | None = None
    embedding_dim: int = 64
    num_layers: int = 4
    num_heads: int = 8
    use_causal_mask: bool = True
    emb_init_scale: float = 0.02
    pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID
    max_sequence_length: int | None = None
    widening_factor: int = 4
    apply_qk_layernorm: bool = False
    apply_post_ln: bool = True
    use_smolgen: bool = True
    smolgen_compress_dim: int = 32
    smolgen_summary_dim: int = 256
    # GNN-specific parameters
    use_gnn: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_message_passing_steps: int = 2

    def __post_init__(self):
        if self.output_size is None:
            self.output_size = self.vocab_size


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


class MultiHeadDotProductAttentionWithGNN(hk.Module):
    """Multi-head attention with GNN-enhanced Smolgen."""

    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int,
        name: str | None = None,
        apply_qk_layernorm: bool = False,
        use_gnn_smolgen: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        smolgen_compress_dim: int = 32,
        smolgen_summary_dim: int = 256,
    ):
        super().__init__(name=name)
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self._apply_qk_layernorm = apply_qk_layernorm
        self._use_gnn_smolgen = use_gnn_smolgen
        
        if self._use_gnn_smolgen:
            self._gnn_smolgen = GNNSmolgenModule(
                num_heads=num_heads,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_num_layers=gnn_num_layers,
                compress_dim=smolgen_compress_dim,
                summary_dim=smolgen_summary_dim,
                name="gnn_smolgen",
            )

    def __call__(
        self,
        inputs_q: jax.Array,
        inputs_kv: jax.Array,
        mask: jax.Array | None = None,
        fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ) -> jax.Array:
        """Multi-head attention with GNN-enhanced position biases."""
        batch_size, sequence_length, embedding_size = inputs_q.shape

        num_hiddens = self._num_hiddens_per_head * self._num_heads
        q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
        k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)

        if self._apply_qk_layernorm:
            q = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(q)
            k = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(k)

        v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
        
        new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
        q = jnp.reshape(q, new_shape)
        k = jnp.reshape(k, new_shape)
        v = jnp.reshape(v, new_shape)

        # Standard dot-product attention
        attention = jnp.einsum('bthd,bThd->bhtT', q, k)
        attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

        if self._use_gnn_smolgen:
            # Add GNN-based positional biases
            supplemental_logits = self._gnn_smolgen(inputs_q, fen_string)
            attention += supplemental_logits
        else:
            # Fallback to standard positional bias
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


# Update the main transformer functions to use GNN attention
def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
    """Creates sinusoidal encodings."""
    freqs = np.arange(0, hidden_size + 1, 2)
    inv_freq = max_timescale ** (-freqs / hidden_size)
    pos_seq = np.arange(start=0, stop=sequence_length)
    sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
    embeddings = np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
    return embeddings[:, :hidden_size]


def embed_sequences(sequences: jax.Array, config: TransformerConfig) -> jax.Array:
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
    """Right-shift sequences by padding on temporal axis."""
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


def _attention_block(
    inputs: jax.Array, 
    config: TransformerConfig,
    fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
) -> jax.Array:
    """GNN-enhanced attention block."""
    batch_size, sequence_length = inputs.shape[:2]
    
    if config.use_causal_mask:
        causal_mask = np.tril(np.ones((batch_size, 1, sequence_length, sequence_length)))
    else:
        causal_mask = None
    
    if config.use_gnn:
        block = MultiHeadDotProductAttentionWithGNN(
            num_heads=config.num_heads,
            num_hiddens_per_head=config.embedding_dim // config.num_heads,
            apply_qk_layernorm=config.apply_qk_layernorm,
            use_gnn_smolgen=True,
            gnn_hidden_dim=config.gnn_hidden_dim,
            gnn_num_layers=config.gnn_num_layers,
            smolgen_compress_dim=config.smolgen_compress_dim,
            smolgen_summary_dim=config.smolgen_summary_dim,
        )
        return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask, fen_string=fen_string)
    else:
        # Fallback to standard attention (you'd need to implement this)
        raise NotImplementedError("Standard attention not implemented in this example")


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
    fen_string: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
) -> jax.Array:
    """GNN-enhanced transformer decoder."""
    inputs = shift_right(targets)
    embeddings = embed_sequences(inputs, config)

    h = embeddings
    for _ in range(config.num_layers):
        attention_input = layer_norm(h)
        attention = _attention_block(attention_input, config, fen_string)
        h += attention

        mlp_input = layer_norm(h)
        mlp_output = _mlp_block(mlp_input, config)
        h += mlp_output

    if config.apply_post_ln:
        h = layer_norm(h)
    
    logits = hk.Linear(config.output_size)(h)
    return jnn.log_softmax(logits, axis=-1)


def build_gnn_transformer_predictor(config: TransformerConfig):
    """Build a GNN-enhanced transformer predictor."""
    model = hk.transform(functools.partial(transformer_decoder, config=config))
    
    # Return the same structure as the original - assuming constants.Predictor is a namedtuple
    # If you have access to constants.Predictor, use that instead
    from collections import namedtuple
    Predictor = namedtuple('Predictor', ['initial_params', 'predict'])
    
    return Predictor(initial_params=model.init, predict=model.apply)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = TransformerConfig(
        vocab_size=1000,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        use_gnn=True,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        max_sequence_length=79,
    )
    
    print("GNN-Enhanced Chess Transformer configured successfully!")
    print(f"Using GNN: {config.use_gnn}")
    print(f"GNN hidden dim: {config.gnn_hidden_dim}")
    print(f"GNN layers: {config.gnn_num_layers}")
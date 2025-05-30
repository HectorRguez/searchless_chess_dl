"""Simplified attention extraction that works directly with the existing engine."""
from collections.abc import Sequence
import os
import json
import functools
import numpy as np
from absl import app
from absl import flags
import chess
import jax
import jax.numpy as jnp
import haiku as hk
from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
import inspect

# Flags for single position analysis
_FEN = flags.DEFINE_string(
    'fen',
    'r2qkb1r/1b1n1ppp/p2ppn2/1p6/3PP3/1QN1BN2/PPP2PPP/R3K2R w KQkq - 0 10',
    'FEN string of the chess position to analyze (default: position after 1.e4).'
)

_AGENT = flags.DEFINE_enum(
    'agent',
    None,
    [
        'local',
        '9M',
        '136M', 
        '270M',
        'stockfish',
        'stockfish_all_moves',
        'leela_chess_zero_depth_1',
        'leela_chess_zero_policy_net',
        'leela_chess_zero_400_sims',
    ],
    'The agent to evaluate.'
)

_ATTENTION_OUTPUT_DIR = flags.DEFINE_string(
    'attention_output_dir',
    './attention_analysis',
    'Directory to save attention analysis results.'
)

def board_to_input_sequence(board: chess.Board) -> jnp.ndarray:
    """Convert chess board to input sequence for the transformer."""
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(jnp.int32)
    return tokenized_fen

def extract_attention_with_builtin_function(engine, board: chess.Board):
    """
    Extract attention using the transformer's built-in attention extraction.
    Much cleaner than monkey patching!
    """
    # Convert board to input
    input_sequence = board_to_input_sequence(board)
    input_batch = jnp.expand_dims(input_sequence, axis=0)
    
    # Import the transformer module
    from searchless_chess.src import transformer
    
    # We need to extract the model parameters and config from the engine
    # Let's try to inspect the engine's predict_fn to get the parameters
    
    # First, let's see if we can determine the engine's configuration
    # by examining the input sequence structure
    seq_len = len(input_sequence)
    print(f"Input sequence length: {seq_len}")
    
    # Based on the transformer code, let's create a reasonable config
    # This should match what the engine was trained with
    config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,  # This should match the tokenizer
        output_size=128,  # Common for action-value models
        embedding_dim=256,  # Reasonable default
        num_layers=8,
        num_heads=8,
        max_sequence_length=seq_len + 2,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        use_causal_mask=False,
        use_smolgen=False,
        apply_post_ln=True,
    )
    
    # Check the actual signature of the predict_fn
    sig = inspect.signature(engine.predict_fn)
    print(f"Engine predict_fn signature: {sig}")
    
    # Since we can't easily extract parameters, let's use a simpler approach
    # We'll modify the transformer module temporarily to capture attention
    captured_attention = []
    
    # Store the original attention block function
    original_attention_block = transformer._attention_block
    
    def patched_attention_block(inputs, config):
        """Patched attention block that captures attention weights."""
        batch_size, sequence_length = inputs.shape[:2]
        if config.use_causal_mask:
            causal_mask = np.tril(
                np.ones((batch_size, 1, sequence_length, sequence_length))
            )
        else:
            causal_mask = None
        
        # Create attention block with extraction enabled
        block = transformer.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            num_hiddens_per_head=config.embedding_dim // config.num_heads,
            apply_qk_layernorm=config.apply_qk_layernorm,
            use_smolgen=config.use_smolgen,
            smolgen_compress_dim=config.smolgen_compress_dim,
            smolgen_summary_dim=config.smolgen_summary_dim,
            extract_attention=True,  # Enable attention extraction
        )
        
        result = block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask)
        
        if isinstance(result, tuple):
            output, attention_weights = result
            captured_attention.append(attention_weights)
            return output
        else:
            return result
    
    # Apply the patch
    transformer._attention_block = patched_attention_block
    
    try:
        # Call the engine's predict function
        if len(sig.parameters) == 1:
            _ = engine.predict_fn(input_batch)
        else:
            rng_key = jax.random.PRNGKey(42)
            _ = engine.predict_fn(input_batch, rng_key)
        
        return captured_attention
        
    finally:
        # Restore the original function
        transformer._attention_block = original_attention_block

def analyze_attention_patterns(attention_weights, board: chess.Board):
    """Analyze the captured attention patterns."""
    analysis = {
        'fen': board.fen(),
        'position_has_e4_piece': board.piece_at(chess.E4) is not None,
        'e4_piece_type': str(board.piece_at(chess.E4)) if board.piece_at(chess.E4) else None,
        'layers': {}
    }
    
    for layer_idx, layer_attention in enumerate(attention_weights):
        batch_size, num_heads, seq_len, _ = layer_attention.shape
        
        layer_analysis = {
            'num_heads': num_heads,
            'sequence_length': seq_len,
            'heads': {}
        }
        
        for head_idx in range(num_heads):
            # Extract attention matrix for this head
            attention_matrix = layer_attention[0, head_idx, :, :]  # Remove batch dimension
            
            head_analysis = {
                'attention_matrix_shape': attention_matrix.shape,
                'max_attention_value': float(jnp.max(attention_matrix)),
                'min_attention_value': float(jnp.min(attention_matrix)),
                'mean_attention_value': float(jnp.mean(attention_matrix)),
                'attention_std': float(jnp.std(attention_matrix)),
                'attention_entropy': float(-jnp.sum(
                    attention_matrix * jnp.log(attention_matrix + 1e-10)
                )),
                # Store the actual attention matrix (convert to list for JSON)
                'attention_matrix': attention_matrix.tolist()
            }
            
            layer_analysis['heads'][f'head_{head_idx}'] = head_analysis
        
        # Layer-level statistics
        all_head_max = [layer_analysis['heads'][f'head_{h}']['max_attention_value'] 
                       for h in range(num_heads)]
        all_head_mean = [layer_analysis['heads'][f'head_{h}']['mean_attention_value'] 
                        for h in range(num_heads)]
        
        layer_analysis['layer_stats'] = {
            'avg_max_attention': float(np.mean(all_head_max)),
            'avg_mean_attention': float(np.mean(all_head_mean)),
            'max_max_attention': float(np.max(all_head_max)),
            'min_max_attention': float(np.min(all_head_max)),
        }
        
        analysis['layers'][f'layer_{layer_idx}'] = layer_analysis
    
    return analysis

def save_attention_analysis(analysis: dict, output_dir: str):
    """Save attention analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on FEN
    fen_short = analysis['fen'].replace('/', '_').replace(' ', '_')[:50]
    output_file = os.path.join(output_dir, f"attention_{fen_short}.json")
    
    # Save full analysis
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Saved attention analysis to {output_file}")
    
    # Also save a summary without the full attention matrices (for readability)
    summary_analysis = {k: v for k, v in analysis.items() if k != 'layers'}
    summary_layers = {}
    
    for layer_name, layer_data in analysis['layers'].items():
        summary_layer = {k: v for k, v in layer_data.items() if k != 'heads'}
        summary_heads = {}
        
        for head_name, head_data in layer_data['heads'].items():
            summary_head = {k: v for k, v in head_data.items() if k != 'attention_matrix'}
            summary_heads[head_name] = summary_head
        
        summary_layer['heads'] = summary_heads
        summary_layers[layer_name] = summary_layer
    
    summary_analysis['layers'] = summary_layers
    
    summary_file = os.path.join(output_dir, f"attention_summary_{fen_short}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_analysis, f, indent=2)
    
    print(f"Saved attention summary to {summary_file}")
    return output_file, summary_file

def print_analysis_summary(analysis: dict):
    """Print a readable summary of the attention analysis."""
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Position: {analysis['fen']}")
    print(f"E4 piece: {analysis['e4_piece_type'] or 'empty'}")
    
    # Overall statistics
    all_max_values = []
    all_mean_values = []
    
    for layer_name, layer_data in analysis['layers'].items():
        stats = layer_data['layer_stats']
        all_max_values.append(stats['avg_max_attention'])
        all_mean_values.append(stats['avg_mean_attention'])
        
        print(f"\n{layer_name.upper()}:")
        print(f"  Avg max attention: {stats['avg_max_attention']:.4f}")
        print(f"  Avg mean attention: {stats['avg_mean_attention']:.4f}")
        print(f"  Highest attention: {stats['max_max_attention']:.4f}")
        print(f"  Sequence length: {layer_data['sequence_length']}")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Average max attention across layers: {np.mean(all_max_values):.4f}")
    print(f"  Average mean attention across layers: {np.mean(all_mean_values):.4f}")
    print(f"  Total layers analyzed: {len(analysis['layers'])}")
    
    # Tokenization info
    tokenized_sequence = tokenizer.tokenize(analysis['fen'])
    print(f"\nTOKENIZATION INFO:")
    print(f"  Sequence length: {len(tokenized_sequence)}")
    print(f"  First 10 tokens: {tokenized_sequence[:10].tolist()}")
    print(f"  Last 10 tokens: {tokenized_sequence[-10:].tolist()}")

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    if not _AGENT.value:
        raise app.UsageError('--agent flag is required')
    
    # Parse FEN string
    try:
        board = chess.Board(_FEN.value)
        print(f"Analyzing position: {_FEN.value}")
        print(f"Board:\n{board}")
        print(f"E4 square contains: {board.piece_at(chess.E4) or 'empty'}")
    except ValueError as e:
        raise app.UsageError(f'Invalid FEN string: {e}')
    
    # Load the engine - no rebuilding needed!
    print(f"Loading {_AGENT.value} engine...")
    engine = constants.ENGINE_BUILDERS[_AGENT.value]()
    
    # Check if this is a neural engine
    if not hasattr(engine, 'predict_fn'):
        print(f"Error: {_AGENT.value} is not a neural engine or doesn't have predict_fn")
        print("This script only works with neural engines that have transformer attention mechanisms.")
        return
    
    print("Neural engine loaded successfully!")
    print(f"Engine type: {type(engine)}")
    
    try:
        # Extract attention patterns
        print("Extracting attention patterns...")
        attention_weights = extract_attention_with_builtin_function(engine, board)
        
        if not attention_weights:
            print("Error: No attention weights were captured")
            return
        
        print(f"Successfully captured attention from {len(attention_weights)} layers")
        
        # Analyze the attention patterns
        print("Analyzing attention patterns...")
        analysis = analyze_attention_patterns(attention_weights, board)
        
        # Save results
        output_file, summary_file = save_attention_analysis(analysis, _ATTENTION_OUTPUT_DIR.value)
        
        # Print summary
        print_analysis_summary(analysis)
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE!")
        print(f"Full results: {output_file}")
        print(f"Summary: {summary_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    app.run(main)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import chess
import re
import argparse
from io import StringIO
import xml.etree.ElementTree as ET
from PIL import Image
import os

# Mapping chess pieces to SVG filenames
SVG_PIECE_MAP = {
    'P': 'wP.svg', 'N': 'wN.svg', 'B': 'wB.svg', 'R': 'wR.svg', 'Q': 'wQ.svg', 'K': 'wK.svg',  # White
    'p': 'bP.svg', 'n': 'bN.svg', 'b': 'bB.svg', 'r': 'bR.svg', 'q': 'bQ.svg', 'k': 'bK.svg'   # Black
}

def load_piece_images(svg_folder='lichess', image_size=128):  # Increased from 64 to 128
    """Load and cache piece images from SVG files."""
    piece_images = {}
    
    print(f"Loading piece images from folder: {svg_folder}")
    
    # Check if folder exists
    if not os.path.exists(svg_folder):
        print(f"Warning: SVG folder '{svg_folder}' not found. Using Unicode fallback.")
        return None
    
    for piece_symbol, svg_filename in SVG_PIECE_MAP.items():
        svg_path = os.path.join(svg_folder, svg_filename)
        
        if os.path.exists(svg_path):
            try:
                # Try multiple methods for SVG loading
                image_loaded = False
                
                # Method 1: Try cairosvg + PIL with explicit dimensions
                try:
                    import cairosvg
                    from PIL import Image
                    import io
                    
                    print(f"Loading {svg_filename} with cairosvg...")
                    
                    # Convert SVG to PNG in memory with explicit size - KEEP TRANSPARENCY
                    png_data = cairosvg.svg2png(
                        url=svg_path, 
                        output_width=image_size, 
                        output_height=image_size,
                        parent_width=image_size,
                        parent_height=image_size
                    )
                    
                    image = Image.open(io.BytesIO(png_data))
                    print(f"Loaded image size: {image.size}, mode: {image.mode}")
                    
                    # Ensure the image is the right size
                    if image.size != (image_size, image_size):
                        print(f"Resizing from {image.size} to {image_size}x{image_size}")
                        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    
                    # KEEP RGBA for transparency - don't convert to RGB
                    if image.mode != 'RGBA':
                        image = image.convert('RGBA')
                    
                    # Convert to numpy array for matplotlib
                    img_array = np.array(image)
                    print(f"Final numpy array shape: {img_array.shape}")
                    
                    piece_images[piece_symbol] = img_array
                    image_loaded = True
                    
                except ImportError:
                    print(f"cairosvg not available, trying alternative methods...")
                except Exception as e:
                    print(f"cairosvg failed for {svg_filename}: {e}")
                
                # Method 2: Try with rsvg (if available)
                if not image_loaded:
                    try:
                        import gi
                        gi.require_version('Rsvg', '2.0')
                        from gi.repository import Rsvg, GdkPixbuf, GLib
                        import cairo
                        
                        print(f"Trying rsvg for {svg_filename}...")
                        
                        handle = Rsvg.Handle()
                        svg_data = open(svg_path, 'rb').read()
                        handle.write(svg_data)
                        handle.close()
                        
                        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, image_size, image_size)
                        ctx = cairo.Context(surface)
                        ctx.scale(image_size/handle.props.width, image_size/handle.props.height)
                        handle.render_cairo(ctx)
                        
                        buf = surface.get_data()
                        img_array = np.ndarray(shape=(image_size, image_size, 4), dtype=np.uint8, buffer=buf)
                        img_array = img_array[:, :, [2, 1, 0, 3]]  # Convert BGRA to RGBA
                        
                        piece_images[piece_symbol] = img_array
                        image_loaded = True
                        
                    except Exception as e:
                        print(f"rsvg method failed for {svg_filename}: {e}")
                
                # Method 3: Try using wand (ImageMagick)
                if not image_loaded:
                    try:
                        from wand.image import Image as WandImage
                        from wand.color import Color
                        
                        print(f"Trying Wand (ImageMagick) for {svg_filename}...")
                        
                        with WandImage(filename=svg_path) as img:
                            img.format = 'png'
                            # DON'T remove alpha channel - keep transparency
                            img.resize(image_size, image_size)
                            
                            # Convert to PIL then numpy - preserve alpha
                            from PIL import Image
                            import io
                            pil_image = Image.open(io.BytesIO(img.make_blob()))
                            if pil_image.size != (image_size, image_size):
                                pil_image = pil_image.resize((image_size, image_size))
                            
                            # Ensure RGBA
                            if pil_image.mode != 'RGBA':
                                pil_image = pil_image.convert('RGBA')
                            
                            piece_images[piece_symbol] = np.array(pil_image)
                            image_loaded = True
                            
                    except ImportError:
                        print("Wand not available")
                    except Exception as e:
                        print(f"Wand method failed for {svg_filename}: {e}")
                
                if image_loaded:
                    final_shape = piece_images[piece_symbol].shape
                    
                    # Verify it's not 1x1
                    if final_shape[0] == 1 or final_shape[1] == 1:
                        print(f"Warning: {svg_filename} loaded as 1x1 - removing from cache")
                        del piece_images[piece_symbol]
                        image_loaded = False
                else:
                    print(f"Failed to load: {svg_filename}")
                    
            except Exception as e:
                print(f"Error loading {svg_path}: {e}")
        else:
            print(f"File not found: {svg_path}")
    
    if piece_images:
        print(f"Successfully loaded {len(piece_images)} piece images")
        return piece_images
    else:
        print("No piece images loaded. Using Unicode fallback.")
        return None

def add_piece_to_plot(ax, piece_symbol, x, y, piece_images, size=0.8):
    """Add a piece image to the matplotlib plot with transparency."""
    if piece_images and piece_symbol in piece_images:
        try:
            # Use SVG image
            image_array = piece_images[piece_symbol]
            
            # Debug: Print image info
            print(f"Adding piece {piece_symbol} at ({x}, {y}), image shape: {image_array.shape}")
            
            # Adjust size based on context - detailed plots get larger pieces
            piece_size = size * 0.9  # Make pieces smaller relative to squares
            
            # Handle RGBA images properly for transparency
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA image - use matplotlib's alpha channel support
                extent = [x-piece_size/2, x+piece_size/2, y-piece_size/2, y+piece_size/2]
                ax.imshow(image_array, extent=extent, zorder=100, interpolation='bilinear')
            else:
                # RGB image - convert to have alpha channel
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # Add alpha channel - make white pixels transparent
                    rgba_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
                    rgba_image[:, :, :3] = image_array
                    
                    # Create alpha mask - make white/light pixels more transparent
                    brightness = np.mean(image_array, axis=2)
                    alpha = 255 - brightness + 50  # Adjust transparency
                    alpha = np.clip(alpha, 0, 255)
                    rgba_image[:, :, 3] = alpha.astype(np.uint8)
                    
                    extent = [x-piece_size/2, x+piece_size/2, y-piece_size/2, y+piece_size/2]
                    ax.imshow(rgba_image, extent=extent, zorder=100, interpolation='bilinear')
                    print(f"Added RGB->RGBA piece {piece_symbol} using imshow method")
                else:
                    print(f"Unexpected image format for {piece_symbol}: {image_array.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error adding piece {piece_symbol} to plot: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to Unicode fallback
    
    # Fallback to Unicode characters (always visible)
    print(f"Using Unicode fallback for piece {piece_symbol} at ({x}, {y})")
    unicode_pieces = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',  # White
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'   # Black
    }
    
    if piece_symbol in unicode_pieces:
        text = ax.text(x, y, unicode_pieces[piece_symbol], 
                      ha='center', va='center', 
                      fontsize=16, fontweight='bold', color='white', 
                      path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')],
                      zorder=200)  # Very high z-order to appear on top
        print(f"Added Unicode text: {text}")
    return False

def get_board_from_fen(fen_string):
    """Parse FEN and return piece positions."""
    if not fen_string:
        return {}
    
    board = chess.Board(fen_string)
    pieces = {}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convert square index to row, col
            file_idx = chess.square_file(square)  # 0-7 (a-h)
            rank_idx = chess.square_rank(square)  # 0-7 (1-8)
            pieces[(rank_idx, file_idx)] = piece.symbol()
    
    return pieces

def parse_attention_log(log_file_path):
    """Parse attention log file and extract attention matrices."""
    attention_data = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse lines like: L4H7_E4_TO_BOARD: [0.002534, ...]
            if '_E4_TO_BOARD:' in line:
                match = re.match(r'L(\d+)H(\d+)_E4_TO_BOARD: \[(.*)\]', line)
                if match:
                    layer, head, values_str = match.groups()
                    values = [float(x.strip()) for x in values_str.split(',')]
                    
                    key = f"L{layer}H{head}"
                    if key not in attention_data:
                        attention_data[key] = {}
                    attention_data[key]['e4_to_board'] = np.array(values)
            
            elif '_BOARD_TO_E4:' in line:
                match = re.match(r'L(\d+)H(\d+)_BOARD_TO_E4: \[(.*)\]', line)
                if match:
                    layer, head, values_str = match.groups()
                    values = [float(x.strip()) for x in values_str.split(',')]
                    
                    key = f"L{layer}H{head}"
                    if key not in attention_data:
                        attention_data[key] = {}
                    attention_data[key]['board_to_e4'] = np.array(values)
    
    return attention_data

def square_index_to_chess_notation(index):
    """Convert 0-63 index to chess notation (a1-h8)."""
    file = chr(ord('a') + (index % 8))
    rank = str((index // 8) + 1)
    return file + rank

def create_chess_board_heatmap(attention_values, title, figsize=(10, 10), 
                               highlight_e4=True, colormap='viridis', fen_string=None, 
                               svg_folder='lichess'):
    """Create a chess board heatmap visualization with piece images."""
    # Load piece images
    piece_images = load_piece_images(svg_folder)
    
    # Reshape 64 values into 8x8 board
    board = attention_values.reshape(8, 8)
    
    # Flip vertically to match chess board orientation (rank 8 at top)
    board = np.flipud(board)
    
    # Get piece positions if FEN provided
    pieces = get_board_from_fen(fen_string) if fen_string else {}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap based on the provided palette (purple to yellow-green)
    if colormap == 'custom' or colormap == 'viridis':
        # Purple to teal to yellow-green gradient like in the image
        colors = ['#440154', '#3b528b', '#21908c', '#5dc863', '#fde725']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=n_bins)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    # Create heatmap with lower z-order so pieces appear on top
    im = ax.imshow(board, cmap=cmap, aspect='equal', vmin=0, vmax=np.max(attention_values), alpha=0.7, zorder=1)
    
    # Add chess board styling
    for i in range(8):
        for j in range(8):
            # Checkerboard pattern background
            square_color = 'lightgray' if (i + j) % 2 == 0 else 'white'
            rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor=square_color, alpha=0.3)
            ax.add_patch(rect)
            
            # Add square labels
            file = chr(ord('a') + j)
            rank = str(8 - i)
            square_name = file + rank
            
            # Get piece for this square (flip coordinates back for piece lookup)
            piece_rank = 7 - i  # Convert display row back to chess rank (0-7)
            piece_file = j      # File stays the same
            piece_symbol = pieces.get((piece_rank, piece_file), '')
            
            # Add piece image only (no text at all)
            if piece_symbol:
                # Use larger size for detailed plots
                piece_added = add_piece_to_plot(ax, piece_symbol, j, i, piece_images, size=1.2)
    
    # Highlight e4 square if requested
    if highlight_e4:
        # E4 is at index 35 in 0-63 array (4*8 + 4 = 36, but 0-indexed so 35)
        # E4 = file e (index 4), rank 4 (index 3) 
        e4_file = 4  # File 'e' is index 4 (a=0, b=1, c=2, d=3, e=4)
        e4_rank = 3  # Rank 4 is index 3 (rank 1=0, rank 2=1, rank 3=2, rank 4=3)
        
        # After flipud(), rank 4 becomes display row: 8-1-3 = 4
        e4_display_row = 8 - 1 - e4_rank  # Flip vertically
        e4_display_col = e4_file
        
        rect = patches.Rectangle((e4_display_col-0.5, e4_display_row-0.5), 1, 1,
                               linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # Set up axes (do this after adding pieces)
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticks(range(8))
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Force a redraw to make sure pieces appear
    plt.draw()
    
    plt.tight_layout()
    return fig, ax

def create_layer_head_grid(attention_data, attention_type='e4_to_board', 
                          figsize=(16, 12), colormap='viridis', fen_string=None, 
                          svg_folder='lichess'):
    """Create a grid of all layer/head combinations like LC0 visualizer."""
    
    # Load piece images once for the entire grid
    piece_images = load_piece_images(svg_folder)
    
    # Extract layer and head numbers
    layers = set()
    heads = set()
    for key in attention_data.keys():
        match = re.match(r'L(\d+)H(\d+)', key)
        if match:
            layer, head = map(int, match.groups())
            layers.add(layer)
            heads.add(head)
    
    max_layer = max(layers)
    max_head = max(heads)
    
    # Get piece positions if FEN provided
    pieces = get_board_from_fen(fen_string) if fen_string else {}
    
    # Create custom viridis-like colormap
    if colormap == 'custom' or colormap == 'viridis':
        colors = ['#440154', '#3b528b', '#21908c', '#5dc863', '#fde725']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=n_bins)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    # Create subplot grid with smaller figure size
    fig, axes = plt.subplots(max_layer + 1, max_head + 1, figsize=figsize)
    
    # Handle single row/column case
    if max_layer == 0:
        axes = axes.reshape(1, -1)
    if max_head == 0:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Attention Patterns: {"E4 → Board" if attention_type == "e4_to_board" else "Board → E4"}',
                fontsize=14, fontweight='bold')
    
    for layer in range(max_layer + 1):
        for head in range(max_head + 1):
            ax = axes[layer, head]
            key = f"L{layer}H{head}"
            
            if key in attention_data and attention_type in attention_data[key]:
                values = attention_data[key][attention_type]
                board = values.reshape(8, 8)
                board = np.flipud(board)  # Flip for chess orientation
                
                # Create heatmap with transparency for pieces
                im = ax.imshow(board, cmap=cmap, aspect='equal', 
                             vmin=0, vmax=np.max(values), alpha=0.8, zorder=1)
                
                # Add checkerboard pattern and pieces
                for i in range(8):
                    for j in range(8):
                        square_color = 'lightgray' if (i + j) % 2 == 0 else 'white'
                        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                               linewidth=0.5, edgecolor='gray',
                                               facecolor=square_color, alpha=0.2, zorder=0)
                        ax.add_patch(rect)
                        
                        # Add pieces with smaller size for grid view
                        piece_rank = 7 - i  # Convert display row back to chess rank
                        piece_file = j
                        piece_symbol = pieces.get((piece_rank, piece_file), '')
                        
                        if piece_symbol:
                            add_piece_to_plot(ax, piece_symbol, j, i, piece_images, size=0.04)
                
                # Highlight e4
                e4_file = 4  # File 'e' 
                e4_rank = 3  # Rank 4 (0-indexed)
                e4_display_row = 8 - 1 - e4_rank  # After flipud()
                e4_display_col = e4_file
                
                rect = patches.Rectangle((e4_display_col-0.5, e4_display_row-0.5), 1, 1,
                                       linewidth=1.5, edgecolor='red', facecolor='none', zorder=50)
                ax.add_patch(rect)
                
                ax.set_title(f'L{layer}H{head}', fontsize=8)
                
            else:
                # Empty subplot for missing data
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(f'L{layer}H{head}', fontsize=8, color='gray')
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def create_top_squares_analysis(attention_data, attention_type='e4_to_board', top_n=5):
    """Create analysis of top attended squares similar to LC0 style."""
    
    analysis = {}
    
    for key, data in attention_data.items():
        if attention_type in data:
            values = data[attention_type]
            
            # Get top N squares
            top_indices = np.argsort(values)[-top_n:][::-1]  # Descending order
            top_squares = []
            
            for idx in top_indices:
                square_name = square_index_to_chess_notation(idx)
                attention_value = values[idx]
                top_squares.append((square_name, attention_value))
            
            analysis[key] = top_squares
    
    return analysis

def print_attention_summary(attention_data, fen_string=None):
    """Print a summary in LC0 visualizer style."""
    
    print("=" * 60)
    print("CHESS ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)
    
    if fen_string:
        print(f"Position: {fen_string}")
        board = chess.Board(fen_string)
        print(f"Board:\n{board}")
    
    print(f"\nTotal layers analyzed: {len(set(k.split('H')[0] for k in attention_data.keys()))}")
    print(f"Total heads analyzed: {len(attention_data)}")
    
    # E4 → Board analysis
    print("\n" + "E4 → BOARD ATTENTION" + "\n" + "-" * 40)
    e4_to_board_analysis = create_top_squares_analysis(attention_data, 'e4_to_board', top_n=3)
    
    for key in sorted(e4_to_board_analysis.keys()):
        top_squares = e4_to_board_analysis[key]
        print(f"{key}: ", end="")
        square_strs = [f"{sq}({val:.3f})" for sq, val in top_squares]
        print(" | ".join(square_strs))
    
    # Board → E4 analysis
    print("\n" + "BOARD → E4 ATTENTION" + "\n" + "-" * 40)
    board_to_e4_analysis = create_top_squares_analysis(attention_data, 'board_to_e4', top_n=3)
    
    for key in sorted(board_to_e4_analysis.keys()):
        top_squares = board_to_e4_analysis[key]
        print(f"{key}: ", end="")
        square_strs = [f"{sq}({val:.3f})" for sq, val in top_squares]
        print(" | ".join(square_strs))

def main():
    parser = argparse.ArgumentParser(description='Visualize chess attention patterns')
    parser.add_argument('log_file', help='Path to attention log file')
    parser.add_argument('--fen', help='FEN string of the position')
    parser.add_argument('--output-dir', default='attention_plots', 
                       help='Output directory for plots')
    parser.add_argument('--svg-folder', default='lichess', 
                       help='Folder containing SVG piece files')
    parser.add_argument('--colormap', default='viridis', 
                       choices=['viridis', 'Reds', 'Blues', 'Greens', 'custom'],
                       help='Colormap for heatmaps')
    
    args = parser.parse_args()
    
    # Parse attention data
    print("Parsing attention log...")
    attention_data = parse_attention_log(args.log_file)
    
    if not attention_data:
        print("No attention data found in log file!")
        return
    
    # Print summary
    print_attention_summary(attention_data, args.fen)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # 1. Grid of all layer/head combinations - E4 to Board
    print("Creating E4 → Board grid...")
    fig = create_layer_head_grid(attention_data, 'e4_to_board', colormap=args.colormap, 
                                fen_string=args.fen, svg_folder=args.svg_folder)
    fig.savefig(f"{args.output_dir}/e4_to_board_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Grid of all layer/head combinations - Board to E4
    print("Creating Board → E4 grid...")
    fig = create_layer_head_grid(attention_data, 'board_to_e4', colormap=args.colormap, 
                                fen_string=args.fen, svg_folder=args.svg_folder)
    fig.savefig(f"{args.output_dir}/board_to_e4_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Individual detailed heatmaps for interesting layers/heads
    print("Creating detailed heatmaps...")
    
    # Find most interesting attention patterns (highest variance or max values)
    interesting_patterns = []
    for key, data in attention_data.items():
        if 'e4_to_board' in data:
            values = data['e4_to_board']
            max_val = np.max(values)
            variance = np.var(values)
            interesting_patterns.append((key, max_val, variance, 'e4_to_board'))
    
    # Sort by max attention value
    interesting_patterns.sort(key=lambda x: x[1], reverse=True)
    
    # Create detailed plots for top 6 most interesting patterns
    for i, (key, max_val, variance, att_type) in enumerate(interesting_patterns[:6]):
        values = attention_data[key][att_type]
        title = f"{key} - E4 → Board (Max: {max_val:.3f}, Var: {variance:.4f})"
        
        fig, ax = create_chess_board_heatmap(values, title, colormap=args.colormap, 
                                           fen_string=args.fen, svg_folder=args.svg_folder)
        fig.savefig(f"{args.output_dir}/detailed_{key}_{att_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\nVisualization complete! Files saved to {args.output_dir}/")
    print("Generated files:")
    print("  - e4_to_board_grid.png (Overview of all layers/heads)")
    print("  - board_to_e4_grid.png (Overview of all layers/heads)")
    print("  - detailed_*.png (Individual heatmaps for most interesting patterns)")

if __name__ == "__main__":
    # Run main if called directly

    main()
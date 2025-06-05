# Simple test script
import matplotlib.pyplot as plt
from your_visualizer import load_piece_images, add_piece_to_plot


def load_piece_images(svg_folder='lichess', image_size=0.8):
    """Load and cache piece images from SVG files."""
    piece_images = {}
    
    for piece_symbol, svg_filename in SVG_PIECE_MAP.items():
        svg_path = os.path.join(svg_folder, svg_filename)
        
        if os.path.exists(svg_path):
            try:
                # For SVG files, we'll use a simple approach
                # You might need to install cairosvg: pip install cairosvg
                try:
                    import cairosvg
                    from PIL import Image
                    import io
                    
                    # Convert SVG to PNG in memory
                    png_data = cairosvg.svg2png(url=svg_path, output_width=image_size, output_height=image_size)
                    image = Image.open(io.BytesIO(png_data))
                    
                    # Convert RGBA to RGB if needed
                    if image.mode == 'RGBA':
                        # Create white background
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                        image = background
                    
                    # Convert to numpy array for matplotlib
                    piece_images[piece_symbol] = np.array(image)
                    image_loaded = True
                 
                    # Method 2: Try matplotlib's svg support (if cairosvg failed)
                    if not image_loaded:
                        try:
                            import matplotlib.image as mpimg
                            from matplotlib.offsetbox import OffsetImage
                            
                            # This won't work directly with SVG, but let's try
                            print(f"Trying matplotlib for {svg_filename}...")
                            
                        except Exception as e:
                            print(f"Matplotlib method failed for {svg_filename}: {e}")
                    
                    # Method 3: Try converting SVG to PNG using other libraries
                    if not image_loaded:
                        try:
                            # Try using wand (ImageMagick)
                            from wand.image import Image as WandImage
                            from wand.color import Color
                            
                            with WandImage(filename=svg_path) as img:
                                img.format = 'png'
                                img.background_color = Color('white')
                                img.alpha_channel = 'remove'
                                img.resize(image_size, image_size)
                                
                                # Convert to PIL then numpy
                                from PIL import Image
                                import io
                                pil_image = Image.open(io.BytesIO(img.make_blob()))
                                piece_images[piece_symbol] = np.array(pil_image)
                                image_loaded = True
                                
                        except ImportError:
                            pass
                        except Exception as e:
                            print(f"Wand method failed for {svg_filename}: {e}")
                    
                    if image_loaded:
                        print(f"Successfully loaded: {svg_filename}")
                    else:
                        print(f"Failed to load: {svg_filename}")
                    
                    
                except ImportError:
                    print(f"Warning: cairosvg not installed. Install with 'pip install cairosvg' to use SVG pieces.")
                    print("Falling back to Unicode characters...")
                    return None
                    
            except Exception as e:
                print(f"Warning: Could not load {svg_path}: {e}")
        else:
            print(f"Warning: SVG file not found: {svg_path}")
    
    return piece_images if piece_images else None

def add_piece_to_plot(ax, piece_symbol, x, y, piece_images, size=0.8):
    """Add a piece image to the matplotlib plot."""
    if piece_images and piece_symbol in piece_images:
        # Use SVG image
        image_array = piece_images[piece_symbol]
        
        # Create OffsetImage
        imagebox = OffsetImage(image_array, zoom=size/64 * 0.8)  # Adjust zoom as needed
        
        # Create AnnotationBbox
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)
        
        return True
    else:
        # Fallback to Unicode characters
        unicode_pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',  # White
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'   # Black
        }
        
        if piece_symbol in unicode_pieces:
            ax.text(x, y - 0.15, unicode_pieces[piece_symbol], ha='center', va='center', 
                   fontsize=24, fontweight='bold')
        return False



fig, ax = plt.subplots()
piece_images = load_piece_images('lichess')
add_piece_to_plot(ax, 'K', 0, 0, piece_images)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.show()
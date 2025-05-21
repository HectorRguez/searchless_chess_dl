# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements tokenization of FEN strings."""
import jaxtyping as jtp
import numpy as np
# pyfmt: disable
_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77

def tokenize(fen: str) -> jtp.Int32[jtp.Array, 'T']:
    """Returns an array of tokens from a fen string.
    
    We compute a tokenized representation of the board, from the FEN string.
    The final array of tokens is a mapping from this string to numbers, which
    are defined in the dictionary `_CHARACTERS_INDEX`.
    
    If it's black's turn to move, we invert the board representation so that
    the current player is always at the bottom (AlphaZero style).
    
    For the 'en passant' information, we convert the '-' (which means there is
    no en passant relevant square) to '..', to always have two characters, and
    a fixed length output.
    
    Args:
      fen: The board position in Forsyth-Edwards Notation.
    """
    # Extracting the relevant information from the FEN.
    board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
    
    # Process the board representation
    indices = []
    
    # Convert FEN board representation to a 2D array (8x8)
    board_2d = []
    ranks = board.split('/')
    
    for rank in ranks:
        rank_squares = []
        for char in rank:
            if char in _SPACES_CHARACTERS:
                # Expand numbers to that many empty squares
                rank_squares.extend(['.'] * int(char))
            else:
                rank_squares.append(char)
        board_2d.append(rank_squares)
    
    # If it's black's turn, invert the board
    is_inverted = (side == 'b')
    if is_inverted:
        # Reverse the ranks (rows)
        board_2d = board_2d[::-1]
        
        # Reverse the files (columns) and swap piece colors
        for i in range(8):
            board_2d[i] = board_2d[i][::-1]
            for j in range(8):
                if board_2d[i][j] != '.':
                    # Swap case: uppercase becomes lowercase and vice versa
                    board_2d[i][j] = board_2d[i][j].lower() if board_2d[i][j].isupper() else board_2d[i][j].upper()
    
    # Flatten the 2D array and convert to tokens
    for rank in board_2d:
        for square in rank:
            indices.append(_CHARACTERS_INDEX[square])
    
    # Process castling rights
    if castling == '-':
        indices.extend(4 * [_CHARACTERS_INDEX['.']])
    else:
        for char in castling:
            indices.append(_CHARACTERS_INDEX[char])
        # Padding castling to have exactly 4 characters.
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])
    
    # Process en passant
    if en_passant == '-':
        indices.extend(2 * [_CHARACTERS_INDEX['.']])
    else:
        # En passant is a square like 'e3'.
        for char in en_passant:
            indices.append(_CHARACTERS_INDEX[char])
    
    # Three digits for halfmoves (since last capture) is enough since the game
    # ends at 50.
    halfmoves_last += '.' * (3 - len(halfmoves_last))
    indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])
    
    # Three digits for full moves is enough (no game lasts longer than 999
    # moves).
    fullmoves += '.' * (3 - len(fullmoves))
    indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

    # Add the turn indicator token at the end
    # You might need to add 'w' and 'b' to your _CHARACTERS list if not already there
    indices.append(_CHARACTERS_INDEX[side])  # Append 'w' or 'b' token
    
    assert len(indices) == SEQUENCE_LENGTH
    return np.asarray(indices, dtype=np.uint8)

def test_fen_tokenizer():
    """Test function to verify the FEN tokenizer's board inversion functionality."""
    # Test cases with their expected behaviors
    test_cases = [
        # Standard starting position
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Starting position, white to move (no inversion)"
        },
        # Same position but black to move - should be inverted
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "description": "Starting position, black to move (board inverted)"
        },
        # Middle game position
        {
            "fen": "r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/2PP1N2/PP3PPP/RNBQKB1R w KQkq - 0 6",
            "description": "Middle game position, white to move (no inversion)"
        },
        # Same middle game position, black to move
        {
            "fen": "r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/2PP1N2/PP3PPP/RNBQKB1R b KQkq - 0 6",
            "description": "Middle game position, black to move (board inverted)"
        }
    ]

    for i, test_case in enumerate(test_cases):
        fen = test_case["fen"]
        description = test_case["description"]
        
        # Tokenize the FEN
        tokens = tokenize(fen)
        
        # Extract the side to move from the FEN
        side = fen.split()[1]
        
        print(f"Test case {i+1}: {description}")
        print(f"FEN: {fen}")
        print(f"Side to move: {'White' if side == 'w' else 'Black'}")
        
        # Convert tokens back to characters for visual inspection
        token_chars = [_CHARACTERS[t] for t in tokens]
        
        # The first 64 tokens represent the 8x8 board
        board_chars = token_chars[:64]
        
        # Print the board in 8x8 format
        print("Board representation in tokens:")
        for rank in range(8):
            rank_str = ''.join(board_chars[rank*8:(rank+1)*8])
            # Replace dots with '.' for better visibility
            rank_str = rank_str.replace('.', '.')
            print(rank_str)
        
        # Print the rest of the tokens (castling, en passant, etc.)
        print("Additional tokens:")
        print(f"Castling: {''.join(token_chars[64:68])}")
        print(f"En passant: {''.join(token_chars[68:70])}")
        print(f"Halfmoves: {''.join(token_chars[70:73])}")
        print(f"Fullmoves: {''.join(token_chars[73:76])}")
        
        # Verify the sequence length
        print(f"Token sequence length: {len(tokens)} (expected: {SEQUENCE_LENGTH})")
        print("-" * 50)

# Run the test
if __name__ == "__main__":
    test_fen_tokenizer()
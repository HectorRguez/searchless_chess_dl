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

"""
Implements a robust, unambiguous tokenization of FEN strings
without board reversal.
"""

import jaxtyping as jtp
import numpy as np

# A single, unambiguous vocabulary for all FEN components.
TOKEN_VOCAB = {
    # Meta tokens
    "meta_pad": 0,
    "meta_w": 1,  # White to move
    "meta_b": 2,  # Black to move
    # Piece tokens (including empty squares)
    "piece_.": 3,
    "piece_p": 4, "piece_n": 5, "piece_b": 6, "piece_r": 7, "piece_q": 8, "piece_k": 9,
    "piece_P": 10, "piece_B": 11, "piece_N": 12, "piece_R": 13, "piece_Q": 14, "piece_K": 15,
    # Castling rights tokens
    "castle_K": 16, "castle_Q": 17, "castle_k": 18, "castle_q": 19,
    # En passant file tokens
    "ep_a": 20, "ep_b": 21, "ep_c": 22, "ep_d": 23, "ep_e": 24, "ep_f": 25,
    "ep_g": 26, "ep_h": 27,
    # En passant rank tokens
    "ep_3": 28, "ep_4": 29, "ep_5": 30, "ep_6": 31,
    # Digit tokens for move counters
    "digit_0": 32, "digit_1": 33, "digit_2": 34, "digit_3": 35, "digit_4": 36,
    "digit_5": 37, "digit_6": 38, "digit_7": 39, "digit_8": 40, "digit_9": 41,
}

# Sequence length: 64 (board) + 4 (castling) + 2 (ep) + 3 (half) + 3 (full) + 1 (side)
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> jtp.UInt8[jtp.Array, "T"]:
	"""Returns a direct, unambiguous array of tokens from a FEN string.

  This tokenizer maps each component of a FEN string to a unique integer
  token. It does NOT perform any board reversal or perspective shifts,
  providing a direct representation of the FEN.

  Args:
    fen: The board position in Forsyth-Edwards Notation.

  Returns:
    A NumPy array of integer tokens.
  """
	board_fen, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")

	tokens = []

	# 1. Tokenize board state (64 tokens)
	for char in board_fen.replace("/", ""):
		if char.isdigit():
			tokens.extend([TOKEN_VOCAB["piece_."]] * int(char))
		else:
			tokens.append(TOKEN_VOCAB[f"piece_{char}"])

	# 2. Tokenize castling rights in a fixed order (4 tokens)
	castle_tokens = []
	if "K" in castling: castle_tokens.append(TOKEN_VOCAB["castle_K"])
	if "Q" in castling: castle_tokens.append(TOKEN_VOCAB["castle_Q"])
	if "k" in castling: castle_tokens.append(TOKEN_VOCAB["castle_k"])
	if "q" in castling: castle_tokens.append(TOKEN_VOCAB["castle_q"])
	castle_tokens.extend([TOKEN_VOCAB["meta_pad"]] * (4 - len(castle_tokens)))
	tokens.extend(castle_tokens)

	# 3. Tokenize en passant square (2 tokens)
	if en_passant == "-":
		tokens.extend([TOKEN_VOCAB["meta_pad"]] * 2)
	else:
		file, rank = en_passant[0], en_passant[1]
		tokens.append(TOKEN_VOCAB[f"ep_{file}"])
		tokens.append(TOKEN_VOCAB[f"ep_{rank}"])

	# 4. Tokenize halfmove clock (3 tokens)
	halfmove_tokens = [TOKEN_VOCAB[f"digit_{d}"] for d in halfmoves]
	halfmove_tokens.extend(
		[TOKEN_VOCAB["meta_pad"]] * (3 - len(halfmove_tokens))
	)
	tokens.extend(halfmove_tokens)

	# 5. Tokenize fullmove number (3 tokens)
	fullmove_tokens = [TOKEN_VOCAB[f"digit_{d}"] for d in fullmoves]
	fullmove_tokens.extend(
		[TOKEN_VOCAB["meta_pad"]] * (3 - len(fullmove_tokens))
	)
	tokens.extend(fullmove_tokens)

	# 6. Add the original side-to-move token (1 token)
	tokens.append(TOKEN_VOCAB[f"meta_{side}"])

	assert len(tokens) == SEQUENCE_LENGTH
	return np.asarray(tokens, dtype=np.uint8)


def test_fen_tokenizer():
	"""Tests that the tokenizer provides a direct, non-canonical mapping."""
	print("--- Testing Tokenizer (Direct, Non-Canonical Representation) ---")

	fen_to_test = "rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R b KQ - 4 8"
	board_fen, side, castling, en_passant, halfmoves, fullmoves = fen_to_test.split(" ")

	print(f"\nTesting FEN: {fen_to_test}")

	# Tokenize the FEN
	tokens = tokenize(fen_to_test)

	# Decode the tokens back into a readable format for verification
	rev_vocab = {v: k for k, v in TOKEN_VOCAB.items()}
	decoded_tokens = [rev_vocab[t] for t in tokens]

	# Extract parts from the decoded tokens
	decoded_board_str = "".join(decoded_tokens[:64]).replace("piece_", "")
	decoded_castling_str = "".join(decoded_tokens[64:68]).replace("castle_", "").replace("meta_pad", "")
	decoded_ep_str = "".join(decoded_tokens[68:70]).replace("ep_", "").replace("meta_pad", "")
	decoded_half_str = "".join(decoded_tokens[70:73]).replace("digit_", "").replace("meta_pad", "")
	decoded_full_str = "".join(decoded_tokens[73:76]).replace("digit_", "").replace("meta_pad", "")
	decoded_side_str = decoded_tokens[76].replace("meta_", "")

	print("\n--- Verification ---")
	print(f"Side to Move:   Original='{side}', Decoded='{decoded_side_str}'")
	print(f"Castling:       Original='{castling}', Decoded='{decoded_castling_str}'")
	print(f"En Passant:     Original='{en_passant}', Decoded='{decoded_ep_str}'")

	# Verify that the decoded tokens match the original FEN components
	assert decoded_side_str == side
	# Note: The decoded castling string will be in canonical order (KQkq)
	assert sorted(decoded_castling_str) == sorted(castling)
	assert decoded_ep_str == en_passant.replace("-", "")
	print("\nSUCCESS: Tokenizer correctly represents the FEN without transformation.")


if __name__ == "__main__":
	test_fen_tokenizer()
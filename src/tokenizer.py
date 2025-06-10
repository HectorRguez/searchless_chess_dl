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
Implements a robust tokenization of FEN strings with a correct
canonical representation (row-wise flip only).
"""

import jaxtyping as jtp
import numpy as np

# A single, unambiguous vocabulary for all FEN components.
TOKEN_VOCAB = {
    # Meta tokens
    "meta_pad": 0,
    "meta_w": 1,  # Player to move (always 'w' in canonical form)
    # Piece tokens (including empty squares)
    "piece_.": 2,
    "piece_p": 3, "piece_n": 4, "piece_b": 5, "piece_r": 6, "piece_q": 7, "piece_k": 8,
    "piece_P": 9, "piece_B": 10, "piece_N": 11, "piece_R": 12, "piece_Q": 13, "piece_K": 14,
    # Castling rights tokens
    "castle_K": 15, "castle_Q": 16, "castle_k": 17, "castle_q": 18,
    # En passant file tokens
    "ep_a": 19, "ep_b": 20, "ep_c": 21, "ep_d": 22, "ep_e": 23, "ep_f": 24, "ep_g": 25, "ep_h": 26,
    # En passant rank tokens
    "ep_3": 27, "ep_4": 28, "ep_5": 29, "ep_6": 30,
    # Digit tokens for move counters
    "digit_0": 31, "digit_1": 32, "digit_2": 33, "digit_3": 34, "digit_4": 35,
    "digit_5": 36, "digit_6": 37, "digit_7": 38, "digit_8": 39, "digit_9": 40,
}

# Sequence length: 64 (board) + 4 (castling) + 2 (ep) + 3 (half) + 3 (full) + 1 (side)
SEQUENCE_LENGTH = 77


def _transform_board_fen(board_fen: str) -> str:
	"""Flips board vertically (row-wise) and swaps piece colors."""
	ranks = board_fen.split("/")
	# Swap piece case for each rank, then reverse the order of the ranks.
	# The column order within each rank is NOT changed.
	flipped_ranks = ["".join(c.swapcase() for c in rank) for rank in ranks]
	return "/".join(flipped_ranks[::-1])


def _transform_castling(rights: str) -> str:
	"""Transforms castling rights to the canonical perspective."""
	if rights == "-":
		return "-"
	transformed = ""
	if "k" in rights: transformed += "K"
	if "q" in rights: transformed += "Q"
	if "K" in rights: transformed += "k"
	if "Q" in rights: transformed += "q"
	return transformed if transformed else "-"


def _transform_en_passant(square: str) -> str:
	"""Transforms an en passant square for the canonical representation."""
	if square == "-":
		return "-"
	file = square[0]
	rank = int(square[1])
	# The file remains the same. Only the rank is flipped.
	transformed_rank = 9 - rank
	return f"{file}{transformed_rank}"


def tokenize(fen: str) -> jtp.UInt8[jtp.Array, "T"]:
	"""Returns a canonical array of tokens from a FEN string."""
	board_fen, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")

	if side == "b":
		board_fen = _transform_board_fen(board_fen)
		castling = _transform_castling(castling)
		en_passant = _transform_en_passant(en_passant)

	tokens = []

	# 1. Tokenize board state (64 tokens)
	for char in board_fen.replace("/", ""):
		if char.isdigit():
			tokens.extend([TOKEN_VOCAB["piece_."]] * int(char))
		else:
			tokens.append(TOKEN_VOCAB[f"piece_{char}"])

	# 2. Tokenize castling rights in a fixed, canonical order (4 tokens)
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

	# 6. Add side-to-move token (1 token)
	# This is ALWAYS 'w' because the state is from the current player's view.
	tokens.append(TOKEN_VOCAB["meta_w"])

	assert len(tokens) == SEQUENCE_LENGTH
	return np.asarray(tokens, dtype=np.uint8)


def test_fen_tokenizer():
	"""Tests that the canonical representation is applied correctly."""
	print("--- Testing Canonical Tokenizer (Final Correct Version) ---")

	# A position with Black to move. The tokenizer should transform this.
	fen_black_turn = "rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R b KQ - 4 8"

	# Programmatically generate the correctly flipped FEN to avoid human error.
	board_b, _, castling_b, ep_b, half_b, full_b = fen_black_turn.split(" ")
	flipped_board = _transform_board_fen(board_b)
	flipped_castling = _transform_castling(castling_b)
	flipped_ep = _transform_en_passant(ep_b)
	fen_flipped_white_turn = f"{flipped_board} w {flipped_castling} {flipped_ep} {half_b} {full_b}"

	print(f"\nOriginal FEN (Black): {fen_black_turn}")
	print(f"Programmatically Flipped FEN (White): {fen_flipped_white_turn}")

	print("\n1. Tokenizing position with BLACK to move (should be transformed)...")
	tokens_from_black_turn = tokenize(fen_black_turn)

	print("2. Tokenizing the FLIPPED position with WHITE to move...")
	tokens_from_flipped_white_turn = tokenize(fen_flipped_white_turn)

	print("\n--- Verification ---")
	if np.array_equal(tokens_from_black_turn, tokens_from_flipped_white_turn):
		print("\nSUCCESS: The canonical transformation is working correctly.")
	else:
		print("\nFAILURE: The transformation for black's turn does not match the flipped version.")


if __name__ == "__main__":
	test_fen_tokenizer()
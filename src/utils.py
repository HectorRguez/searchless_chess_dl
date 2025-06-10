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

"""Implements some utility functions."""

import math

import chess
import numpy as np


# The lists of the strings of the row and columns of a chess board,
# traditionally named rank and file.
_CHESS_FILE = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _compute_all_possible_actions() -> tuple[dict[str, int], dict[int, str]]:
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol("Q"))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol("N"))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [("2", "1"), ("7", "8")]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f"{file}{rank}{file}{next_rank}"
      promotion_moves += [(move + piece) for piece in ["q", "r", "b", "n"]]

      # Capture promotions.
      # Left side.
      if file > "a":
        next_file = _CHESS_FILE[index_file - 1]
        move = f"{file}{rank}{next_file}{next_rank}"
        promotion_moves += [(move + piece) for piece in ["q", "r", "b", "n"]]
      # Right side.
      if file < "h":
        next_file = _CHESS_FILE[index_file + 1]
        move = f"{file}{rank}{next_file}{next_rank}"
        promotion_moves += [(move + piece) for piece in ["q", "r", "b", "n"]]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
NUM_ACTIONS = len(MOVE_TO_ACTION)


# NEW FUNCTION TO SUPPORT CANONICAL REPRESENTATION
def transform_uci_move(uci_move: str) -> str:
	"""Transforms a UCI move string to its canonical equivalent.

  This is used when the board is flipped for black's turn. It performs a
  row-wise flip of the move's coordinates. The file (column) is unchanged.

  Examples:
    'e7e5' becomes 'e2e4'
    'g8f6' becomes 'g1f3'
    'a7a8q' becomes 'a2a1q' (promotion is preserved)

  Args:
    uci_move: The standard UCI move string.

  Returns:
    The transformed UCI move string.
  """
	from_file = uci_move[0]
	from_rank = int(uci_move[1])
	to_file = uci_move[2]
	to_rank = int(uci_move[3])
	promotion = uci_move[4:] if len(uci_move) > 4 else ""

	# The file does not change, only the rank is flipped from 1-8 to 8-1.
	new_from_rank = 9 - from_rank
	new_to_rank = 9 - to_rank

	return f"{from_file}{new_from_rank}{to_file}{new_to_rank}{promotion}"


def centipawns_to_win_probability(centipawns: int) -> float:
	"""Returns the win probability (in [0, 1]) from the centipawn score."""
	return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * centipawns)) - 1)


def get_uniform_buckets_edges_values(
    num_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Returns edges and values of uniformly sampled buckets in [0, 1]."""
	full_linspace = np.linspace(0.0, 1.0, num_buckets + 1)
	edges = full_linspace[1:-1]
	values = (full_linspace[:-1] + full_linspace[1:]) / 2
	return edges, values


def compute_return_buckets_from_returns(
    returns: np.ndarray,
    bins_edges: np.ndarray,
) -> np.ndarray:
	"""Arranges the discounted returns into bins."""
	if len(returns.shape) != 1:
		raise ValueError(
			"The passed returns should be of rank 1. Got"
			f" rank={len(returns.shape)}."
		)
	if len(bins_edges.shape) != 1:
		raise ValueError(
			"The passed bins_edges should be of rank 1. Got"
			f" rank{len(bins_edges.shape)}."
		)
	return np.searchsorted(bins_edges, returns, side="left")


# NEW TEST FUNCTION
def test_move_transformation():
	"""Tests that the UCI move transformation works correctly."""
	print("--- Testing UCI Move Transformation ---")
	test_cases = {
		"e7e5": "e2e4",  # Black's pawn push -> White's pawn push
		"g8f6": "g1f3",  # Black's knight move -> White's knight move
		"d1h5": "d8h4",  # A white queen move
		"a7a8q": "a2a1q",  # Black promoting to a queen
		"e1g1": "e8g8",  # White castling -> Black castling
	}
	all_passed = True
	for original, expected in test_cases.items():
		transformed = transform_uci_move(original)
		if transformed == expected:
			print(f"SUCCESS: '{original}' -> '{transformed}'")
		else:
			print(
				f"FAILURE: '{original}' -> '{transformed}', expected"
				f" '{expected}'"
			)
			all_passed = False
	print(
		"\nMove transformation test PASSED."
		if all_passed
		else "\nMove transformation test FAILED."
	)


if __name__ == "__main__":
	test_move_transformation()
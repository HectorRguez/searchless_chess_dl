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
"""Evaluates engines on the puzzles dataset from lichess with ELO estimation."""

from collections.abc import Sequence
import io
import os
import json
import numpy as np
from typing import List, Dict
from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import pandas as pd
from scipy.optimize import minimize_scalar

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib

_NUM_PUZZLES = flags.DEFINE_integer(
    name='num_puzzles',
    default=None,
    help='The number of puzzles to evaluate.',
    required=True,
)

_AGENT = flags.DEFINE_enum(
    name='agent',
    default=None,
    enum_values=[
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
    help='The agent to evaluate.',
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    name='output_path',
    default='puzzle_results.json',
    help='Path to save the results and ELO estimation.',
)


class SimpleBayesElo:
    """Simple MLE-based ELO estimator for puzzle performance."""
    
    def __init__(self):
        self.results = []  # 1.0 = solved, 0.0 = failed
        self.puzzle_ratings = []  # Known puzzle ratings
        self.detailed_results = []  # Store all puzzle details
    
    def add_result(self, puzzle_rating: int, solved: bool, puzzle_id: int = None):
        """Add a single puzzle result with optional metadata."""
        self.results.append(1.0 if solved else 0.0)
        self.puzzle_ratings.append(puzzle_rating)
        self.detailed_results.append({
            'puzzle_id': puzzle_id,
            'rating': puzzle_rating,
            'solved': solved,
        })
    
    def estimate_elo(self) -> Dict:
        """
        Estimate engine ELO using Maximum Likelihood Estimation.
        Uses standard ELO formula: P(win) = 1 / (1 + 10^((opponent_elo - player_elo)/400))
        """
        if len(self.results) == 0:
            return {"error": "No results"}
        
        def negative_log_likelihood(engine_elo):
            """Negative log-likelihood function to minimize."""
            ll = 0.0
            for result, puzzle_rating in zip(self.results, self.puzzle_ratings):
                # Probability engine solves puzzle
                prob = 1.0 / (1.0 + 10**((puzzle_rating - engine_elo) / 400.0))
                # Add to log likelihood (with small epsilon to avoid log(0))
                if result > 0.5:  # Solved
                    ll += np.log(max(prob, 1e-15))
                else:  # Failed
                    ll += np.log(max(1 - prob, 1e-15))
            return -ll  # Return negative for minimization
        
        # Find ELO that maximizes likelihood
        result = minimize_scalar(
            negative_log_likelihood, 
            bounds=(600, 3500), 
            method='bounded'
        )
        
        estimated_elo = result.x
        
        # Calculate standard error using Fisher Information approximation
        fisher_info = 0.0
        for res, puzzle_rating in zip(self.results, self.puzzle_ratings):
            prob = 1.0 / (1.0 + 10**((puzzle_rating - estimated_elo) / 400.0))
            # Fisher Information derivative term
            derivative = prob * (1 - prob) * (np.log(10) / 400.0)
            if prob > 1e-10 and (1 - prob) > 1e-10:  # Avoid division by near-zero
                fisher_info += derivative**2 / (prob * (1 - prob))
        
        std_error = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else 100.0
        
        # Performance statistics
        total_puzzles = len(self.results)
        solved_puzzles = int(sum(self.results))
        success_rate = solved_puzzles / total_puzzles
        avg_puzzle_rating = np.mean(self.puzzle_ratings)
        
        # Performance by rating bins
        rating_bins = self._analyze_by_rating_bins()
        
        return {
            'estimated_elo': round(estimated_elo, 1),
            'std_error': round(std_error, 1),
            'confidence_95_lower': round(estimated_elo - 1.96 * std_error, 1),
            'confidence_95_upper': round(estimated_elo + 1.96 * std_error, 1),
            'total_puzzles': total_puzzles,
            'solved_puzzles': solved_puzzles,
            'success_rate': round(success_rate, 3),
            'avg_puzzle_rating': round(avg_puzzle_rating, 1),
            'min_puzzle_rating': min(self.puzzle_ratings),
            'max_puzzle_rating': max(self.puzzle_ratings),
            'rating_bins': rating_bins,
            'log_likelihood': -result.fun  # Convert back to positive log-likelihood
        }
    
    def _analyze_by_rating_bins(self, bin_size: int = 200) -> Dict:
        """Analyze performance by puzzle rating bins."""
        if len(self.results) == 0:
            return {}
        
        min_rating = min(self.puzzle_ratings)
        max_rating = max(self.puzzle_ratings)
        
        # Create bins
        bins = []
        current_bin = (min_rating // bin_size) * bin_size
        while current_bin <= max_rating:
            bins.append((current_bin, current_bin + bin_size))
            current_bin += bin_size
        
        bin_analysis = {}
        for bin_start, bin_end in bins:
            bin_results = []
            bin_ratings = []
            
            for rating, result in zip(self.puzzle_ratings, self.results):
                if bin_start <= rating < bin_end:
                    bin_results.append(result)
                    bin_ratings.append(rating)
            
            if bin_results:
                bin_analysis[f"{bin_start}-{bin_end-1}"] = {
                    'count': len(bin_results),
                    'solved': int(sum(bin_results)),
                    'success_rate': round(np.mean(bin_results), 3),
                    'avg_rating': round(np.mean(bin_ratings), 1)
                }
        
        return bin_analysis
    
    def get_summary_report(self, agent_name: str) -> str:
        """Generate a formatted summary report."""
        results = self.estimate_elo()
        
        if 'error' in results:
            return f"Error: {results['error']}"
        
        report = f"""
        Engine: {agent_name:<54}
        Estimated ELO: {results['estimated_elo']:<8} ± {results['std_error']:<8}
        95% Confidence Interval: [{results['confidence_95_lower']:<6}, {results['confidence_95_upper']:<6}]
        Puzzles solved: {results['solved_puzzles']:<4}/{results['total_puzzles']:<4} ({results['success_rate']:.1%})
        Average puzzle rating: {results['avg_puzzle_rating']:<8}
        Puzzle rating range: {results['min_puzzle_rating']:<4} - {results['max_puzzle_rating']:<4}
        """
        
        for bin_name, bin_data in results['rating_bins'].items():
            report += f"{bin_name:>12}: {bin_data['solved']:>3}/{bin_data['count']:<3} ({bin_data['success_rate']:>5.1%}) avg={bin_data['avg_rating']:>6.1f}\n"
        
        return report


def evaluate_puzzle_from_pandas_row(
    puzzle: pd.Series,
    engine: engine_lib.Engine,
) -> bool:
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
    if game is None:
        raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
    board = game.end().board()
    return evaluate_puzzle_from_board(
        board=board,
        moves=puzzle['Moves'].split(' '),
        engine=engine,
    )


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine: engine_lib.Engine,
) -> bool:
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    for move_idx, move in enumerate(moves):
        # According to https://database.lichess.org/#puzzles, the FEN is the
        # position before the opponent makes their move. The position to present to
        # the player is after applying the first move to that FEN. The second move
        # is the beginning of the solution.
        if move_idx % 2 == 1:
            predicted_move = engine.play(board=board).uci()
            # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
            # check if the `predicted_move` results in a checkmate if it differs from
            # the solution.
            if move != predicted_move:
                board.push(chess.Move.from_uci(predicted_move))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(move))
    return True


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    puzzles_path = '/data/hector/searchless_chess/puzzles.csv'

    
    print(f"Loading puzzles from: {puzzles_path}")
    puzzles = pd.read_csv(puzzles_path, nrows=_NUM_PUZZLES.value)
    engine = constants.ENGINE_BUILDERS[_AGENT.value]()
    
    # Initialize ELO estimator
    elo_estimator = SimpleBayesElo()
    
    print(f"\n🔍 Evaluating {_AGENT.value} on {len(puzzles)} puzzles...")
    print("=" * 80)
    
    # Evaluate puzzles
    for puzzle_id, puzzle in puzzles.iterrows():
        try:
            correct = evaluate_puzzle_from_pandas_row(
                puzzle=puzzle,
                engine=engine,
            )
            
            # Add result to ELO estimator
            elo_estimator.add_result(
                puzzle_rating=puzzle['Rating'], 
                solved=correct,
                puzzle_id=int(puzzle_id)
            )
            
            # Print progress
            status = "✅" if correct else "❌"
            print(f"Puzzle {puzzle_id:4d}: {status} Rating: {puzzle['Rating']:4d}")
            
            # Show intermediate ELO estimate every 100 puzzles
            if (puzzle_id + 1) % 100 == 0:
                temp_results = elo_estimator.estimate_elo()
                if 'estimated_elo' in temp_results:
                    print(f"\n📊 Intermediate estimate after {puzzle_id + 1} puzzles:")
                    print(f"   ELO: {temp_results['estimated_elo']:.0f} ± {temp_results['std_error']:.0f}")
                    print(f"   Success rate: {temp_results['success_rate']:.1%}")
                    print("-" * 60)
        
        except Exception as e:
            print(f"❌ Error evaluating puzzle {puzzle_id}: {e}")
            continue
    
    # Generate final report
    print("\n" + "=" * 80)
    print("🏆 FINAL ELO ESTIMATION RESULTS")
    print("=" * 80)
    
    final_results = elo_estimator.estimate_elo()
    
    if 'error' not in final_results:
        print(elo_estimator.get_summary_report(_AGENT.value))
        
        # Additional analysis
        print(f"\n📈 Statistical Information:")
        print(f"   Log-likelihood: {final_results['log_likelihood']:.2f}")
        print(f"   Sample size: {final_results['total_puzzles']}")
        
        expected_performance = []
        for rating in elo_estimator.puzzle_ratings:
            prob = 1.0 / (1.0 + 10**((rating - final_results['estimated_elo']) / 400.0))
            expected_performance.append(prob)
        
        print(f"   Expected success rate: {np.mean(expected_performance):.1%}")
        print(f"   Actual success rate: {final_results['success_rate']:.1%}")
        
        # Save detailed results
        output_data = {
            'agent': _AGENT.value,
            'num_puzzles': _NUM_PUZZLES.value,
            'elo_estimation': final_results,
            'detailed_results': elo_estimator.detailed_results,
            'methodology': 'Maximum Likelihood Estimation using ELO logistic model'
        }
        
        with open(_OUTPUT_PATH.value, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {_OUTPUT_PATH.value}")
        
    else:
        print(f"❌ Error in ELO estimation: {final_results['error']}")

if __name__ == '__main__':
    # Run main evaluation
    app.run(main)
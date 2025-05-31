"""
Simple Bayesian ELO estimation for chess puzzle performance.
Only needs: results (solved/failed) and puzzle ratings.
"""

import numpy as np
from typing import List, Dict
from scipy.optimize import minimize_scalar


class SimpleBayesElo:
    """Minimal Bayesian ELO estimator for puzzle performance."""
    
    def __init__(self):
        self.results = []  # 1.0 = solved, 0.0 = failed
        self.puzzle_ratings = []  # Known puzzle ratings
    
    def add_result(self, puzzle_rating: int, solved: bool):
        """Add a single puzzle result."""
        self.results.append(1.0 if solved else 0.0)
        self.puzzle_ratings.append(puzzle_rating)
    
    def estimate_elo(self) -> Dict:
        """
        Estimate engine ELO using Maximum Likelihood Estimation.
        
        Uses standard ELO formula: P(win) = 1 / (1 + 10^((opponent_elo - player_elo)/400))
        """
        if len(self.results) == 0:
            return {"error": "No results"}
        
        # Maximum Likelihood Estimation
        def negative_log_likelihood(engine_elo):
            ll = 0.0
            for result, puzzle_rating in zip(self.results, self.puzzle_ratings):
                # Probability engine solves puzzle
                prob = 1.0 / (1.0 + 10**((puzzle_rating - engine_elo) / 400.0))
                # Add to log likelihood
                if result > 0.5:  # Solved
                    ll += np.log(prob + 1e-15)  # Small epsilon to avoid log(0)
                else:  # Failed
                    ll += np.log(1 - prob + 1e-15)
            return -ll  # Return negative for minimization
        
        # Find ELO that maximizes likelihood
        result = minimize_scalar(
            negative_log_likelihood, 
            bounds=(600, 3500), 
            method='bounded'
        )
        
        estimated_elo = result.x
        
        # Calculate standard error using Fisher Information
        fisher_info = 0.0
        for res, puzzle_rating in zip(self.results, self.puzzle_ratings):
            prob = 1.0 / (1.0 + 10**((puzzle_rating - estimated_elo) / 400.0))
            # Fisher Information = sum of prob * (1-prob) * (d_prob/d_elo)^2
            derivative = prob * (1 - prob) * (np.log(10) / 400.0)
            fisher_info += derivative**2 / (prob * (1 - prob))
        
        std_error = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else 100.0
        
        # Calculate performance statistics
        total_puzzles = len(self.results)
        solved_puzzles = int(sum(self.results))
        success_rate = solved_puzzles / total_puzzles
        avg_puzzle_rating = np.mean(self.puzzle_ratings)
        
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
            'max_puzzle_rating': max(self.puzzle_ratings)
        }
    
    def get_results_summary(self) -> str:
        """Get a formatted summary of the ELO estimation."""
        results = self.estimate_elo()
        
        if 'error' in results:
            return f"Error: {results['error']}"
        
        return f"""
Bayesian ELO Estimation Results:
================================
Estimated ELO: {results['estimated_elo']} ± {results['std_error']}
95% Confidence Interval: [{results['confidence_95_lower']}, {results['confidence_95_upper']}]

Performance Summary:
- Puzzles solved: {results['solved_puzzles']}/{results['total_puzzles']} ({results['success_rate']:.1%})
- Average puzzle rating: {results['avg_puzzle_rating']}
- Puzzle rating range: {results['min_puzzle_rating']} - {results['max_puzzle_rating']}
"""


# Integration with your existing puzzle evaluation code
def main_with_simple_bayes_elo(argv):
    """Modified main function using simple Bayesian ELO."""
    
    # ... your existing setup code ...
    puzzles_path = "data/hector/searchless_chess/puzzles.csv"
    puzzles = pd.read_csv(puzzles_path, nrows=_NUM_PUZZLES.value)
    engine = constants.ENGINE_BUILDERS[_AGENT.value]()
    
    # Initialize simple Bayes ELO estimator
    elo_estimator = SimpleBayesElo()
    
    print(f"Evaluating {_AGENT.value} on {len(puzzles)} puzzles...")
    
    for puzzle_id, puzzle in puzzles.iterrows():
        # Evaluate puzzle (your existing code)
        correct = evaluate_puzzle_from_pandas_row(puzzle=puzzle, engine=engine)
        
        # Add result to ELO estimator (this is all we need!)
        elo_estimator.add_result(puzzle['Rating'], correct)
        
        # Print progress
        print(f"Puzzle {puzzle_id}: {'✓' if correct else '✗'} (rating: {puzzle['Rating']})")
        
        # Show intermediate ELO estimate every 100 puzzles
        if (puzzle_id + 1) % 100 == 0:
            temp_results = elo_estimator.estimate_elo()
            print(f"\nAfter {puzzle_id + 1} puzzles:")
            print(f"  Estimated ELO: {temp_results['estimated_elo']} ± {temp_results['std_error']}")
            print(f"  Success rate: {temp_results['success_rate']:.1%}")
            print("-" * 50)
    
    # Final results
    print("\n" + "="*60)
    print("FINAL BAYESIAN ELO ANALYSIS")
    print("="*60)
    print(elo_estimator.get_results_summary())


# Example usage with just results data
if __name__ == "__main__":
    # Example: if you already have results
    estimator = SimpleBayesElo()
    
    # Add some example results (in practice, these come from your puzzle evaluation)
    example_results = [
        (1200, True),   # Solved 1200-rated puzzle
        (1500, True),   # Solved 1500-rated puzzle  
        (1800, False),  # Failed 1800-rated puzzle
        (2000, False),  # Failed 2000-rated puzzle
        # ... etc
    ]
    
    for rating, solved in example_results:
        estimator.add_result(rating, solved)
    
    print(estimator.get_results_summary())
#!/usr/bin/env python3
"""
Experiment Runner - For testing different algorithm parameters
This allows you to easily compare different configurations
"""

import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from main import NewsStockAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Run multiple experiments with different parameters
    to find the best configuration
    """

    def __init__(self, base_config_path: str = 'config.yaml'):
        """Initialize experiment runner"""
        self.base_config_path = base_config_path
        self.results_dir = Path('experiments/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_single_experiment(
        self,
        name: str,
        weights: Dict = None,
        scoring: Dict = None,
        description: str = ""
    ) -> Dict:
        """
        Run a single experiment with custom parameters

        Args:
            name: Experiment name
            weights: Custom weight configuration
            scoring: Custom scoring configuration
            description: Description of what's being tested

        Returns:
            Experiment results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Experiment: {name}")
        logger.info(f"{'='*60}")
        if description:
            logger.info(f"Description: {description}")

        # Create custom config
        custom_config = {}
        if weights:
            custom_config['weights'] = weights
            logger.info(f"Custom Weights: {weights}")
        if scoring:
            custom_config['scoring'] = scoring
            logger.info(f"Custom Scoring: {scoring}")

        # Initialize analyzer
        analyzer = NewsStockAnalyzer(config_path=self.base_config_path)

        # Run experiment
        results, metrics = analyzer.run_experiment(name, custom_config)

        # Save detailed results
        experiment_data = {
            'name': name,
            'description': description,
            'timestamp': datetime.utcnow().isoformat(),
            'custom_config': custom_config,
            'metrics': metrics,
            'results': {
                symbol: {
                    'signal': result['signal'],
                    'correlation': result['correlation'] if 'error' not in result['correlation'] else None
                }
                for symbol, result in results.items()
            }
        }

        # Save to file
        result_file = self.results_dir / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        logger.info(f"Results saved to: {result_file}")

        return experiment_data

    def run_weight_experiments(self):
        """
        Run experiments with different weight configurations
        to find optimal balance
        """
        logger.info("Running Weight Optimization Experiments...")

        experiments = [
            {
                'name': 'baseline',
                'description': 'Baseline configuration from config.yaml',
                'weights': None
            },
            {
                'name': 'sentiment_heavy',
                'description': 'Prioritize sentiment over other factors',
                'weights': {
                    'sentiment_score': 0.50,
                    'entity_relevance': 0.20,
                    'source_diversity': 0.15,
                    'volume_spike': 0.10,
                    'framing_consistency': 0.05
                }
            },
            {
                'name': 'entity_heavy',
                'description': 'Prioritize entity relevance',
                'weights': {
                    'sentiment_score': 0.25,
                    'entity_relevance': 0.40,
                    'source_diversity': 0.15,
                    'volume_spike': 0.15,
                    'framing_consistency': 0.05
                }
            },
            {
                'name': 'balanced',
                'description': 'Equal weighting of all factors',
                'weights': {
                    'sentiment_score': 0.20,
                    'entity_relevance': 0.20,
                    'source_diversity': 0.20,
                    'volume_spike': 0.20,
                    'framing_consistency': 0.20
                }
            },
            {
                'name': 'volume_focused',
                'description': 'Focus on volume spikes as indicator',
                'weights': {
                    'sentiment_score': 0.25,
                    'entity_relevance': 0.20,
                    'source_diversity': 0.10,
                    'volume_spike': 0.35,
                    'framing_consistency': 0.10
                }
            }
        ]

        results = []
        for exp in experiments:
            result = self.run_single_experiment(
                name=exp['name'],
                weights=exp['weights'],
                description=exp['description']
            )
            results.append(result)

        # Compare results
        self._compare_experiments(results)

        return results

    def run_sentiment_multiplier_experiments(self):
        """
        Test different sentiment multipliers to see which
        captures market impact better
        """
        logger.info("Running Sentiment Multiplier Experiments...")

        experiments = [
            {
                'name': 'conservative_sentiment',
                'description': 'Lower multipliers - conservative approach',
                'scoring': {
                    'positive_sentiment_multiplier': 1.0,
                    'negative_sentiment_multiplier': 1.2
                }
            },
            {
                'name': 'aggressive_sentiment',
                'description': 'Higher multipliers - aggressive approach',
                'scoring': {
                    'positive_sentiment_multiplier': 1.5,
                    'negative_sentiment_multiplier': 2.0
                }
            },
            {
                'name': 'negative_bias',
                'description': 'Emphasize negative news more',
                'scoring': {
                    'positive_sentiment_multiplier': 1.0,
                    'negative_sentiment_multiplier': 2.5
                }
            }
        ]

        results = []
        for exp in experiments:
            result = self.run_single_experiment(
                name=exp['name'],
                scoring=exp['scoring'],
                description=exp['description']
            )
            results.append(result)

        self._compare_experiments(results)

        return results

    def _compare_experiments(self, results: List[Dict]):
        """Compare multiple experiment results"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPARISON")
        logger.info("="*60)

        print(f"\n{'Experiment':<25} {'Avg Corr':<12} {'Avg Conf':<12} {'BUY':<6} {'SELL':<6} {'HOLD':<6}")
        print("-" * 75)

        for result in results:
            metrics = result['metrics']
            signals = metrics['signals']

            print(f"{result['name']:<25} "
                  f"{metrics['avg_correlation']:<12.3f} "
                  f"{metrics['avg_confidence']:<12.3f} "
                  f"{signals['BUY']:<6} "
                  f"{signals['SELL']:<6} "
                  f"{signals['HOLD']:<6}")

        # Find best performer
        best = max(results, key=lambda x: x['metrics']['avg_correlation'])
        logger.info(f"\nBest Performer: {best['name']}")
        logger.info(f"  Average Correlation: {best['metrics']['avg_correlation']:.3f}")
        logger.info(f"  Description: {best['description']}")

    def create_custom_experiment(self, name: str, config: Dict, description: str = ""):
        """
        Create a custom experiment with your own parameters

        Example usage:
            runner.create_custom_experiment(
                name='my_experiment',
                config={
                    'weights': {
                        'sentiment_score': 0.40,
                        'entity_relevance': 0.30,
                        ...
                    },
                    'scoring': {
                        'positive_sentiment_multiplier': 1.3,
                        ...
                    }
                },
                description='Testing my hypothesis'
            )
        """
        return self.run_single_experiment(
            name=name,
            weights=config.get('weights'),
            scoring=config.get('scoring'),
            description=description
        )


def main():
    """Run predefined experiment suites"""
    import argparse

    parser = argparse.ArgumentParser(description='Run algorithm experiments')
    parser.add_argument('--suite', choices=['weights', 'sentiment', 'all'],
                       default='weights', help='Experiment suite to run')
    parser.add_argument('--custom', help='Path to custom experiment config JSON')

    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.custom:
        # Load custom experiment config
        with open(args.custom, 'r') as f:
            custom_config = json.load(f)

        runner.create_custom_experiment(
            name=custom_config.get('name', 'custom'),
            config=custom_config.get('config', {}),
            description=custom_config.get('description', '')
        )

    elif args.suite == 'weights':
        runner.run_weight_experiments()

    elif args.suite == 'sentiment':
        runner.run_sentiment_multiplier_experiments()

    elif args.suite == 'all':
        runner.run_weight_experiments()
        runner.run_sentiment_multiplier_experiments()


if __name__ == "__main__":
    main()

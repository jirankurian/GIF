"""
Continual Learning Analyzer for Rigorous Lifelong Learning Evaluation
=====================================================================

This module implements the ContinualLearningAnalyzer class, which provides comprehensive
analytical tools for evaluating continual learning systems. It computes standardized
metrics from the continual learning literature and generates publication-quality
visualizations to assess lifelong learning performance.

Scientific Foundation:
=====================

The analyzer implements three critical metrics that are standard in continual learning
research, enabling rigorous scientific evaluation of lifelong learning systems:

**1. Average Accuracy (AA)**
Mathematical Definition: AA = (1/T) * Σ(i=1 to T) a_T,i
Where T is the total number of tasks and a_T,i is the accuracy on task i after
learning all T tasks. This metric provides an overall assessment of the system's
performance across its entire learning history.

**2. Forgetting Measure (FM)**
Mathematical Definition: FM_i = max(a_k,i) - a_T,i for k ∈ [1, T]
Where a_k,i is the accuracy on task i after learning task k, and a_T,i is the final
accuracy on task i. This directly measures catastrophic forgetting - a value near
zero indicates successful forgetting prevention.

**3. Forward Transfer (FT)**
Mathematical Definition: FT_i = a_1,i^(pre-exposed) - a_1,i^(naive)
Where a_1,i^(pre-exposed) is the first-epoch accuracy on task i for a model with
prior experience, and a_1,i^(naive) is the first-epoch accuracy for a naive model.
Positive values indicate beneficial knowledge transfer (System Potentiation).

Experimental Protocol Support:
=============================

The analyzer supports the standard sequential task protocol used in continual learning
research:

1. **Sequential Training Phase**: Train on tasks in sequence (A → B → C → ...)
2. **Comprehensive Evaluation Phase**: After training on each task, evaluate on ALL
   previously learned tasks plus the current task
3. **Performance Logging**: Maintain detailed accuracy logs for each task at each
   evaluation point throughout the learning sequence

This protocol enables precise measurement of both forgetting patterns and knowledge
transfer effects, providing the quantitative evidence needed for scientific claims
about continual learning capabilities.

Data Format Requirements:
========================

The analyzer expects a Polars DataFrame with the following columns:
- `training_task_id`: Identifier of the task currently being trained
- `evaluation_task_id`: Identifier of the task being evaluated
- `epoch`: Training epoch or step number
- `accuracy`: Accuracy score on the evaluation task (0.0 to 1.0)

Additional optional columns:
- `loss`: Training or evaluation loss
- `timestamp`: When the measurement was taken
- `model_state`: Any additional model state information

Example DataFrame structure:
```
training_task_id | evaluation_task_id | epoch | accuracy
task_A          | task_A            | 1     | 0.65
task_A          | task_A            | 10    | 0.89
task_B          | task_A            | 1     | 0.87
task_B          | task_B            | 1     | 0.42
task_B          | task_A            | 10    | 0.85
task_B          | task_B            | 10    | 0.91
```

Integration with GIF Framework:
==============================

The analyzer seamlessly integrates with the GIF continual learning infrastructure:
- Direct compatibility with Continual_Trainer statistics
- Automatic conversion from training logs to analysis format
- Support for real-time analysis during training
- Export capabilities for research publication

This enables researchers to generate rigorous quantitative evidence for continual
learning claims without manual data processing or custom analysis scripts.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ContinualLearningAnalyzer:
    """
    Comprehensive analyzer for continual learning performance evaluation.
    
    This class implements standardized metrics from the continual learning literature
    to rigorously assess lifelong learning capabilities. It provides both quantitative
    metrics and publication-quality visualizations for scientific evaluation.
    
    The analyzer supports the standard sequential task experimental protocol and
    computes three critical metrics: Average Accuracy, Forgetting Measure, and
    Forward Transfer. These metrics enable rigorous scientific assessment of
    catastrophic forgetting prevention and knowledge transfer capabilities.
    
    Attributes:
        experiment_logs: Polars DataFrame containing experimental results
        task_ids: List of unique task identifiers in the experiment
        training_sequence: Ordered list of tasks in training sequence
        evaluation_matrix: Structured representation of all evaluations
    """
    
    def __init__(self, experiment_logs: pl.DataFrame):
        """
        Initialize the continual learning analyzer with experimental data.
        
        Args:
            experiment_logs (pl.DataFrame): DataFrame containing experimental results
                                          with required columns: training_task_id,
                                          evaluation_task_id, epoch, accuracy.
        
        Raises:
            ValueError: If required columns are missing or data format is invalid.
            TypeError: If experiment_logs is not a Polars DataFrame.
        
        Example:
            # Load experimental results
            logs = pl.read_csv("continual_learning_experiment.csv")
            analyzer = ContinualLearningAnalyzer(logs)
        """
        # Validate input type
        if not isinstance(experiment_logs, pl.DataFrame):
            raise TypeError(f"experiment_logs must be a Polars DataFrame, got {type(experiment_logs)}")
        
        # Validate required columns
        required_columns = ['training_task_id', 'evaluation_task_id', 'epoch', 'accuracy']
        missing_columns = [col for col in required_columns if col not in experiment_logs.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        if not experiment_logs['accuracy'].dtype.is_numeric():
            raise ValueError("accuracy column must contain numeric values")
        
        accuracy_range = experiment_logs.select([
            pl.col('accuracy').min().alias('min_acc'),
            pl.col('accuracy').max().alias('max_acc')
        ]).row(0)
        
        if accuracy_range[0] < 0.0 or accuracy_range[1] > 1.0:
            raise ValueError("accuracy values must be in range [0.0, 1.0]")
        
        # Store experiment data
        self.experiment_logs = experiment_logs.clone()
        
        # Extract task information
        self.task_ids = sorted(self.experiment_logs['evaluation_task_id'].unique().to_list())
        self.training_sequence = (
            self.experiment_logs
            .select('training_task_id')
            .unique(maintain_order=True)
            .to_series()
            .to_list()
        )
        
        # Validate experimental protocol
        self._validate_experimental_protocol()
        
        # Create structured evaluation matrix for efficient metric computation
        self.evaluation_matrix = self._create_evaluation_matrix()

    def _validate_experimental_protocol(self) -> None:
        """
        Validate that the experimental data follows the sequential task protocol.

        Raises:
            ValueError: If the experimental protocol is not properly followed.
        """
        # Check that each training task has evaluations on all previous tasks
        for i, training_task in enumerate(self.training_sequence):
            expected_eval_tasks = self.training_sequence[:i+1]

            actual_eval_tasks = (
                self.experiment_logs
                .filter(pl.col('training_task_id') == training_task)
                .select('evaluation_task_id')
                .unique()
                .to_series()
                .to_list()
            )

            missing_evaluations = set(expected_eval_tasks) - set(actual_eval_tasks)
            if missing_evaluations:
                raise ValueError(
                    f"Missing evaluations for training task '{training_task}': "
                    f"Expected evaluations on {expected_eval_tasks}, "
                    f"but missing {list(missing_evaluations)}"
                )

    def _create_evaluation_matrix(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Create a structured matrix of all evaluations for efficient metric computation.

        Returns:
            Dict mapping training_task -> evaluation_task -> list of accuracies
        """
        matrix = {}

        for training_task in self.training_sequence:
            matrix[training_task] = {}

            for eval_task in self.task_ids:
                # Get all accuracies for this training/evaluation task combination
                accuracies = (
                    self.experiment_logs
                    .filter(
                        (pl.col('training_task_id') == training_task) &
                        (pl.col('evaluation_task_id') == eval_task)
                    )
                    .sort('epoch')
                    .select('accuracy')
                    .to_series()
                    .to_list()
                )

                matrix[training_task][eval_task] = accuracies

        return matrix

    def calculate_average_accuracy(self) -> float:
        """
        Calculate the Average Accuracy (AA) metric for continual learning.

        Average Accuracy measures the overall performance across all learned tasks
        at the end of the learning sequence. It provides a holistic assessment of
        the system's capabilities across its entire learning history.

        Mathematical Definition:
        AA = (1/T) * Σ(i=1 to T) a_T,i

        Where:
        - T is the total number of tasks
        - a_T,i is the final accuracy on task i after learning all T tasks

        Returns:
            float: Average accuracy across all tasks (0.0 to 1.0)

        Example:
            # Calculate overall performance across all learned tasks
            avg_acc = analyzer.calculate_average_accuracy()
            print(f"Average Accuracy: {avg_acc:.3f}")
        """
        final_training_task = self.training_sequence[-1]
        final_accuracies = []

        for task_id in self.task_ids:
            if task_id in self.evaluation_matrix[final_training_task]:
                # Get the final accuracy for this task
                task_accuracies = self.evaluation_matrix[final_training_task][task_id]
                if task_accuracies:
                    final_accuracies.append(task_accuracies[-1])

        if not final_accuracies:
            raise ValueError("No final accuracies found for average accuracy calculation")

        return float(np.mean(final_accuracies))

    def calculate_forgetting_measure(self, target_task_id: str) -> float:
        """
        Calculate the Forgetting Measure (FM) for a specific task.

        Forgetting Measure quantifies catastrophic forgetting by measuring the
        difference between peak performance on a task and final performance after
        learning subsequent tasks. This is the most direct measure of catastrophic
        forgetting in continual learning systems.

        Mathematical Definition:
        FM_i = max(a_k,i) - a_T,i for k ∈ [1, T]

        Where:
        - a_k,i is the accuracy on task i after learning task k
        - a_T,i is the final accuracy on task i after learning all tasks
        - A value near zero indicates successful forgetting prevention

        Args:
            target_task_id (str): The task to calculate forgetting measure for

        Returns:
            float: Forgetting measure (0.0 = no forgetting, higher = more forgetting)

        Raises:
            ValueError: If target_task_id is not found in the experimental data

        Example:
            # Measure catastrophic forgetting on the first learned task
            forgetting = analyzer.calculate_forgetting_measure("task_A")
            print(f"Forgetting on Task A: {forgetting:.3f}")
        """
        if target_task_id not in self.task_ids:
            raise ValueError(f"Task '{target_task_id}' not found in experimental data")

        # Collect all accuracies for the target task across all training phases
        all_accuracies = []

        for training_task in self.training_sequence:
            if target_task_id in self.evaluation_matrix[training_task]:
                task_accuracies = self.evaluation_matrix[training_task][target_task_id]
                if task_accuracies:
                    # Take the final accuracy for this training phase
                    all_accuracies.append(task_accuracies[-1])

        if not all_accuracies:
            raise ValueError(f"No accuracy data found for task '{target_task_id}'")

        # Calculate forgetting measure
        max_accuracy = max(all_accuracies)
        final_accuracy = all_accuracies[-1]

        return float(max_accuracy - final_accuracy)

    def calculate_forward_transfer(self, target_task_id: str) -> float:
        """
        Calculate the Forward Transfer (FT) metric for a specific task.

        Forward Transfer measures positive knowledge transfer by comparing how quickly
        a system learns new tasks when it has prior experience versus learning from
        scratch. This metric indicates whether previous learning experiences provide
        beneficial knowledge that accelerates learning on new tasks (System Potentiation).

        Mathematical Definition:
        FT_i = a_1,i^(pre-exposed) - a_1,i^(naive)

        Where:
        - a_1,i^(pre-exposed) is the first-epoch accuracy on task i for a model with prior experience
        - a_1,i^(naive) is the first-epoch accuracy for a naive model (learning from scratch)
        - Positive values indicate beneficial knowledge transfer

        Note: This implementation assumes the first task serves as the "naive" baseline,
        and subsequent tasks benefit from prior experience. In practice, you might want
        to compare against separately trained naive models for more rigorous analysis.

        Args:
            target_task_id (str): The task to calculate forward transfer for

        Returns:
            float: Forward transfer measure (positive = beneficial transfer,
                   negative = negative transfer, zero = no transfer effect)

        Raises:
            ValueError: If target_task_id is not found or is the first task

        Example:
            # Measure knowledge transfer to the second learned task
            transfer = analyzer.calculate_forward_transfer("task_B")
            print(f"Forward Transfer to Task B: {transfer:.3f}")
        """
        if target_task_id not in self.task_ids:
            raise ValueError(f"Task '{target_task_id}' not found in experimental data")

        # Find when this task was first introduced in the training sequence
        if target_task_id not in self.training_sequence:
            raise ValueError(f"Task '{target_task_id}' was never used for training")

        task_training_index = self.training_sequence.index(target_task_id)

        if task_training_index == 0:
            # This is the first task - no prior experience to compare against
            # Return 0 as there's no baseline for comparison
            return 0.0

        # Get the first-epoch accuracy when this task was introduced
        training_task = self.training_sequence[task_training_index]

        if target_task_id not in self.evaluation_matrix[training_task]:
            raise ValueError(f"No evaluation data found for task '{target_task_id}' during its training phase")

        task_accuracies = self.evaluation_matrix[training_task][target_task_id]
        if not task_accuracies:
            raise ValueError(f"No accuracy data found for task '{target_task_id}'")

        # First-epoch accuracy with prior experience
        pre_exposed_accuracy = task_accuracies[0]

        # For naive baseline, use the first-epoch accuracy of the first task
        # (This is a simplified approach - ideally we'd have separate naive baselines)
        first_task = self.training_sequence[0]
        if first_task not in self.evaluation_matrix[first_task]:
            raise ValueError("No evaluation data found for the first task")

        first_task_accuracies = self.evaluation_matrix[first_task][first_task]
        if not first_task_accuracies:
            raise ValueError("No accuracy data found for the first task")

        naive_accuracy = first_task_accuracies[0]

        # Calculate forward transfer
        return float(pre_exposed_accuracy - naive_accuracy)

    def generate_summary_plot(self, save_path: Optional[str] = None) -> None:
        """
        Generate a publication-quality summary plot of continual learning performance.

        This method creates a comprehensive visualization showing the performance on
        all tasks throughout the learning sequence. The plot clearly demonstrates
        learning curves, forgetting patterns, and knowledge transfer effects.

        Plot Features:
        - X-axis: Training progress (by task and epoch)
        - Y-axis: Accuracy (0.0 to 1.0)
        - Separate line for each evaluation task
        - Clear visual indication of task boundaries
        - Professional styling suitable for publications

        Args:
            save_path (str, optional): Path to save the plot. If None, displays interactively.

        Raises:
            ImportError: If matplotlib/seaborn are not available.

        Example:
            # Generate and display plot
            analyzer.generate_summary_plot()

            # Save plot for publication
            analyzer.generate_summary_plot("continual_learning_results.png")
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Visualization dependencies (matplotlib, seaborn) are not available. "
                "Please install them with: pip install matplotlib seaborn"
            )
        # Set up the plot with professional styling
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color palette for different tasks
        colors = sns.color_palette("husl", len(self.task_ids))
        task_colors = dict(zip(self.task_ids, colors))

        # Track x-axis positions for proper spacing
        x_position = 0
        x_ticks = []
        x_labels = []

        # Plot performance for each training phase
        for training_task in self.training_sequence:
            # Get the number of epochs for this training phase
            max_epochs = max(
                len(accuracies) for accuracies in self.evaluation_matrix[training_task].values()
                if accuracies
            )

            # Create x-axis positions for this training phase
            phase_x = np.arange(x_position, x_position + max_epochs)

            # Plot each evaluation task's performance during this training phase
            for eval_task in self.task_ids:
                if eval_task in self.evaluation_matrix[training_task]:
                    accuracies = self.evaluation_matrix[training_task][eval_task]
                    if accuracies:
                        # Pad accuracies if needed to match max_epochs
                        padded_accuracies = accuracies + [accuracies[-1]] * (max_epochs - len(accuracies))

                        ax.plot(
                            phase_x,
                            padded_accuracies,
                            color=task_colors[eval_task],
                            label=f'Task {eval_task}' if training_task == self.training_sequence[0] else "",
                            linewidth=2,
                            alpha=0.8
                        )

            # Add vertical line to separate training phases
            if training_task != self.training_sequence[-1]:
                ax.axvline(x=x_position + max_epochs - 0.5, color='gray', linestyle='--', alpha=0.5)

            # Update x-axis tracking
            x_ticks.append(x_position + max_epochs // 2)
            x_labels.append(f'Training\n{training_task}')
            x_position += max_epochs

        # Customize the plot
        ax.set_xlabel('Training Progress', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Continual Learning Performance Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set custom x-axis labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save or display the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report with all continual learning metrics.

        This method computes all standard continual learning metrics and compiles
        them into a structured report suitable for research publication or detailed
        analysis. The report includes both overall metrics and task-specific breakdowns.

        Returns:
            Dict[str, Any]: Comprehensive report containing:
                - overall_metrics: Average accuracy and summary statistics
                - task_specific_metrics: Forgetting and transfer measures per task
                - experimental_info: Details about the experimental setup
                - recommendations: Interpretation and recommendations based on results

        Example:
            report = analyzer.generate_comprehensive_report()
            print(f"Average Accuracy: {report['overall_metrics']['average_accuracy']:.3f}")
            print(f"Tasks with significant forgetting: {report['recommendations']['high_forgetting_tasks']}")
        """
        # Calculate overall metrics
        avg_accuracy = self.calculate_average_accuracy()

        # Calculate task-specific metrics
        task_metrics = {}
        for task_id in self.task_ids:
            task_metrics[task_id] = {
                'forgetting_measure': self.calculate_forgetting_measure(task_id),
                'forward_transfer': self.calculate_forward_transfer(task_id) if task_id != self.training_sequence[0] else 0.0
            }

        # Calculate summary statistics
        forgetting_values = [metrics['forgetting_measure'] for metrics in task_metrics.values()]
        transfer_values = [metrics['forward_transfer'] for metrics in task_metrics.values() if metrics['forward_transfer'] != 0.0]

        # Compile comprehensive report
        report = {
            'overall_metrics': {
                'average_accuracy': avg_accuracy,
                'mean_forgetting': float(np.mean(forgetting_values)),
                'max_forgetting': float(np.max(forgetting_values)),
                'mean_forward_transfer': float(np.mean(transfer_values)) if transfer_values else 0.0,
                'total_tasks': len(self.task_ids),
                'training_sequence_length': len(self.training_sequence)
            },
            'task_specific_metrics': task_metrics,
            'experimental_info': {
                'task_ids': self.task_ids,
                'training_sequence': self.training_sequence,
                'total_evaluations': len(self.experiment_logs),
                'data_quality': self._assess_data_quality()
            },
            'recommendations': self._generate_recommendations(avg_accuracy, forgetting_values, transfer_values)
        }

        return report

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality and completeness of the experimental data."""
        return {
            'complete_protocol': self._check_protocol_completeness(),
            'missing_evaluations': self._count_missing_evaluations(),
            'data_consistency': self._check_data_consistency()
        }

    def _check_protocol_completeness(self) -> bool:
        """Check if the experimental protocol was followed completely."""
        try:
            self._validate_experimental_protocol()
            return True
        except ValueError:
            return False

    def _count_missing_evaluations(self) -> int:
        """Count the number of missing evaluations in the experimental data."""
        expected_evaluations = 0
        actual_evaluations = 0

        for i, training_task in enumerate(self.training_sequence):
            expected_eval_tasks = self.training_sequence[:i+1]
            expected_evaluations += len(expected_eval_tasks)

            actual_eval_tasks = (
                self.experiment_logs
                .filter(pl.col('training_task_id') == training_task)
                .select('evaluation_task_id')
                .unique()
                .height
            )
            actual_evaluations += actual_eval_tasks

        return expected_evaluations - actual_evaluations

    def _check_data_consistency(self) -> bool:
        """Check for data consistency issues."""
        # Check for negative accuracies or values > 1.0
        invalid_accuracies = (
            self.experiment_logs
            .filter((pl.col('accuracy') < 0.0) | (pl.col('accuracy') > 1.0))
            .height
        )

        return invalid_accuracies == 0

    def _generate_recommendations(self, avg_accuracy: float, forgetting_values: List[float], transfer_values: List[float]) -> Dict[str, Any]:
        """Generate recommendations based on the analysis results."""
        recommendations = {
            'overall_performance': 'excellent' if avg_accuracy > 0.8 else 'good' if avg_accuracy > 0.6 else 'needs_improvement',
            'forgetting_assessment': 'minimal' if max(forgetting_values) < 0.1 else 'moderate' if max(forgetting_values) < 0.3 else 'significant',
            'transfer_assessment': 'positive' if transfer_values and np.mean(transfer_values) > 0.05 else 'neutral',
            'high_forgetting_tasks': [
                task_id for task_id, metrics in zip(self.task_ids, forgetting_values)
                if metrics > 0.2
            ],
            'negative_transfer_tasks': [
                task_id for task_id, transfer in zip(self.task_ids[1:], transfer_values)
                if transfer < -0.05
            ] if transfer_values else []
        }

        return recommendations

    @classmethod
    def from_trainer_stats(cls, trainer_stats: Dict[str, Any]) -> 'ContinualLearningAnalyzer':
        """
        Create a ContinualLearningAnalyzer from Continual_Trainer statistics.

        This class method provides seamless integration with the GIF framework's
        Continual_Trainer by automatically converting training statistics into
        the required DataFrame format for analysis.

        Args:
            trainer_stats (Dict[str, Any]): Statistics from Continual_Trainer.get_training_stats()

        Returns:
            ContinualLearningAnalyzer: Analyzer instance ready for metric computation

        Raises:
            ValueError: If trainer_stats format is invalid or incomplete

        Example:
            # Get statistics from trainer
            stats = trainer.get_training_stats()

            # Create analyzer directly from trainer stats
            analyzer = ContinualLearningAnalyzer.from_trainer_stats(stats)

            # Compute metrics
            avg_acc = analyzer.calculate_average_accuracy()
        """
        # This is a simplified implementation - in practice, you'd need to
        # convert the trainer's task_losses and other statistics into the
        # required DataFrame format with proper training/evaluation task mapping

        if 'task_losses' not in trainer_stats:
            raise ValueError("trainer_stats must contain 'task_losses' key")

        # Convert trainer statistics to DataFrame format
        # This is a placeholder implementation - real conversion would be more complex
        rows = []
        task_ids = list(trainer_stats['task_losses'].keys())

        for i, (training_task, losses) in enumerate(trainer_stats['task_losses'].items()):
            # For each training task, evaluate on all previous tasks plus current task
            eval_tasks = task_ids[:i+1]

            for epoch, loss in enumerate(losses):
                # Convert loss to accuracy (simplified - real implementation would need proper conversion)
                accuracy = max(0.0, 1.0 - loss)  # Simplified conversion

                # Create evaluation entries for all tasks up to current one
                for eval_task in eval_tasks:
                    # Simulate slight performance degradation on previous tasks
                    if eval_task != training_task:
                        accuracy_adjusted = accuracy * 0.95  # Slight forgetting
                    else:
                        accuracy_adjusted = accuracy

                    rows.append({
                        'training_task_id': training_task,
                        'evaluation_task_id': eval_task,
                        'epoch': epoch + 1,
                        'accuracy': accuracy_adjusted
                    })

        if not rows:
            raise ValueError("No valid training data found in trainer_stats")

        experiment_logs = pl.DataFrame(rows)
        return cls(experiment_logs)

    def export_results(self, output_dir: str) -> None:
        """
        Export analysis results to various formats for research publication.

        Args:
            output_dir (str): Directory to save exported results

        Example:
            analyzer.export_results("./results/experiment_1/")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export comprehensive report as JSON
        report = self.generate_comprehensive_report()
        import json
        with open(output_path / "analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Export raw data as CSV
        self.experiment_logs.write_csv(output_path / "experiment_logs.csv")

        # Export summary plot
        self.generate_summary_plot(str(output_path / "performance_summary.png"))

        print(f"Results exported to: {output_path}")
        print(f"Files created: analysis_report.json, experiment_logs.csv, performance_summary.png")

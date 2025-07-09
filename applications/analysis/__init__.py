"""
Analysis Module for the General Intelligence Framework
=====================================================

This module provides comprehensive analytical tools for evaluating the performance
of continual learning systems within the GIF framework. It implements standardized
metrics and visualization tools that enable rigorous scientific assessment of
lifelong learning capabilities.

The Science of Continual Learning Evaluation:
============================================

In the field of continual (lifelong) learning, researchers have developed standardized
metrics that go beyond simple accuracy measurements. These metrics are essential for
evaluating whether a system truly learns continuously without catastrophic forgetting:

**Average Accuracy**: Measures overall performance across all learned tasks at the
end of the learning sequence. This provides a holistic view of the system's
capabilities across its entire learning history.

**Forgetting Measure**: Quantifies catastrophic forgetting by measuring the difference
between peak performance on a task and final performance after learning subsequent
tasks. A value near zero indicates successful forgetting prevention.

**Forward Transfer**: Measures positive knowledge transfer by comparing how quickly
a system learns new tasks when it has prior experience versus learning from scratch.
Positive values indicate beneficial transfer (System Potentiation).

Experimental Protocol:
=====================

The analysis tools support the standard sequential task protocol used in continual
learning research:

1. **Sequential Training**: Train on Task A, then Task B, then Task C, etc.
2. **Comprehensive Evaluation**: After training on each new task, evaluate performance
   on ALL previously learned tasks plus the current task.
3. **Performance Tracking**: Maintain detailed logs of accuracy on each task at each
   evaluation point throughout the learning sequence.

This protocol enables measurement of both forgetting (performance degradation on old
tasks) and transfer (improved learning on new tasks due to prior experience).

Key Components:
==============

**ContinualLearningAnalyzer**: The main analysis class that computes standard continual
learning metrics from experimental logs. Accepts Polars DataFrames containing training
and evaluation results from sequential task experiments.

**Visualization Tools**: Publication-quality plotting functions that create clear
visualizations of continual learning performance, including learning curves for
each task and forgetting patterns over time.

**Integration Utilities**: Helper functions that seamlessly integrate with the
Continual_Trainer class, enabling automatic analysis of training results without
manual data conversion.

Example Usage:
=============

    import polars as pl
    from applications.analysis.continual_learning_analyzer import ContinualLearningAnalyzer
    
    # Load experimental results from sequential task training
    experiment_logs = pl.read_csv("experiment_results.csv")
    
    # Create analyzer instance
    analyzer = ContinualLearningAnalyzer(experiment_logs)
    
    # Calculate standard continual learning metrics
    avg_accuracy = analyzer.calculate_average_accuracy()
    forgetting_task_a = analyzer.calculate_forgetting_measure("task_A")
    transfer_task_b = analyzer.calculate_forward_transfer("task_B")
    
    # Generate publication-quality visualizations
    analyzer.generate_summary_plot()
    
    # Create comprehensive analysis report
    report = analyzer.generate_comprehensive_report()
    
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Forgetting on Task A: {forgetting_task_a:.3f}")
    print(f"Forward Transfer to Task B: {transfer_task_b:.3f}")

Integration with GIF Framework:
==============================

The analysis module seamlessly integrates with all GIF components:

- **Continual_Trainer**: Automatic conversion of training statistics to analysis format
- **Experimental Protocols**: Support for standardized continual learning experiments
- **Research Publication**: Export capabilities for scientific papers and presentations
- **Collaborative Research**: Standardized metrics enable comparison across studies

This analysis infrastructure enables the GIF framework to provide rigorous scientific
evidence for its continual learning capabilities, supporting claims about catastrophic
forgetting prevention and positive knowledge transfer.
"""

from .continual_learning_analyzer import ContinualLearningAnalyzer

__all__ = ['ContinualLearningAnalyzer']

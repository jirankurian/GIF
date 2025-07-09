"""
Training Module for the General Intelligence Framework
=====================================================

This module implements the training infrastructure for continual learning within
the GIF framework. It provides sophisticated training algorithms that enable the
system to learn continuously from new data without catastrophically forgetting
previously acquired knowledge.

Key Components:
==============

**Continual_Trainer**: The core training class that implements Gradient Episodic
Memory (GEM) algorithm for preventing catastrophic forgetting. This trainer
orchestrates the complete learning cycle including experience storage, gradient
computation, interference detection, and gradient projection.

**GEM Algorithm**: A state-of-the-art continual learning algorithm that prevents
catastrophic forgetting by constraining new learning to not interfere with
previously learned tasks. The algorithm works by:

1. Computing gradients for new learning tasks
2. Sampling past experiences from episodic memory
3. Computing gradients for past experiences
4. Detecting interference through gradient dot products
5. Projecting new gradients to avoid negative interference
6. Applying the projected gradients to update model parameters

Biological Inspiration:
======================

The training algorithms in this module draw inspiration from biological learning
mechanisms, particularly the role of the hippocampus in preventing catastrophic
forgetting through:

- **Experience Replay**: Random reactivation of past experiences during learning
- **Gradient Projection**: Constraint-based learning that preserves existing knowledge
- **Episodic Memory**: Structured storage and retrieval of past learning experiences

Integration with GIF Framework:
==============================

The training module seamlessly integrates with all other GIF components:

- **DU Core**: Utilizes the injected plasticity rules and memory systems
- **RTL Mechanisms**: Leverages synaptic plasticity rules for weight updates
- **Episodic Memory**: Samples past experiences for continual learning constraints
- **GIF Orchestrator**: Works with the complete cognitive cycle pipeline

Example Usage:
=============

    from gif_framework.training.trainer import Continual_Trainer
    from gif_framework.core.rtl_mechanisms import STDP_Rule
    from gif_framework.core.memory_systems import EpisodicMemory
    from gif_framework.core.du_core import DU_Core_V1
    from gif_framework.orchestrator import GIF
    import torch.optim as optim
    import torch.nn as nn
    
    # Create DU Core with continual learning components
    plasticity_rule = STDP_Rule(
        learning_rate_ltp=0.01,
        learning_rate_ltd=0.005,
        tau_ltp=20.0,
        tau_ltd=20.0
    )
    memory = EpisodicMemory(capacity=10000)
    
    du_core = DU_Core_V1(
        input_size=100,
        hidden_sizes=[64, 32],
        output_size=10,
        plasticity_rule=plasticity_rule,
        memory_system=memory
    )
    
    # Create GIF orchestrator
    gif = GIF(du_core)
    gif.attach_encoder(MyEncoder())
    gif.attach_decoder(MyDecoder())
    
    # Create continual learning trainer
    optimizer = optim.Adam(gif.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainer = Continual_Trainer(gif, optimizer, criterion)
    
    # Train with continual learning
    for data, target in training_data:
        loss = trainer.train_step(data, target, task_id="task_1")
        print(f"Training loss: {loss:.4f}")

This training infrastructure enables the GIF framework to achieve true continual
learning capabilities, making it fundamentally more robust and brain-like than
traditional AI systems that suffer from catastrophic forgetting.
"""

from .trainer import Continual_Trainer

__all__ = ['Continual_Trainer']

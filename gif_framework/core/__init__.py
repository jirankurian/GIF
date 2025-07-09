"""
GIF Core Module
===============

Contains the core components of the General Intelligence Framework:
- DU Core v1: The original LIF-based spiking neural network core
- DU Core v2: Advanced hybrid SNN/SSM architecture with attention mechanisms
- RTL Mechanisms: Real-time learning plasticity rules
- Memory Systems: Episodic memory and experience storage
"""

# Import core components for easy access
from .du_core import DU_Core_V1
from .du_core_v2 import DU_Core_V2, HybridSNNSSMLayer
from .rtl_mechanisms import (
    PlasticityRuleInterface,
    STDP_Rule,
    ThreeFactor_Hebbian_Rule
)
from .memory_systems import (
    ExperienceTuple,
    EpisodicMemory
)

# Import knowledge augmentation (with graceful fallback)
try:
    from .knowledge_augmenter import KnowledgeAugmenter
except ImportError:
    # Graceful fallback if knowledge dependencies are not installed
    KnowledgeAugmenter = None

__all__ = [
    # DU Core architectures
    'DU_Core_V1',
    'DU_Core_V2',
    'HybridSNNSSMLayer',

    # RTL mechanisms
    'PlasticityRuleInterface',
    'STDP_Rule',
    'ThreeFactor_Hebbian_Rule',

    # Memory systems
    'ExperienceTuple',
    'EpisodicMemory',

    # Knowledge augmentation
    'KnowledgeAugmenter'
]

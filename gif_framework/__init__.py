"""
General Intelligence Framework (GIF)
====================================

A modular, extensible platform for building artificial general intelligence systems
based on spiking neural networks and real-time learning mechanisms.

The GIF framework provides:
- Modular encoder/decoder interfaces for plug-and-play functionality
- Deep Understanding (DU) Core with real-time learning capabilities
- Episodic memory systems for continuous learning
- Hardware simulation capabilities for neuromorphic computing

Author: GIF Development Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "GIF Development Team"

# Import core components
from .orchestrator import GIF, MetaController
from .interfaces.base_interfaces import EncoderInterface, DecoderInterface, SpikeTrain, Action

# Import meta-cognitive components (with graceful fallback)
try:
    from .module_library import ModuleLibrary, ModuleMetadata, ModuleEntry
except ImportError:
    ModuleLibrary = None
    ModuleMetadata = None
    ModuleEntry = None

__all__ = [
    # Core orchestration
    'GIF',
    'MetaController',

    # Interfaces
    'EncoderInterface',
    'DecoderInterface',
    'SpikeTrain',
    'Action',

    # Meta-cognitive components
    'ModuleLibrary',
    'ModuleMetadata',
    'ModuleEntry'
]

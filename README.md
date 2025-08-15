# General Intelligence Framework (GIF)

A landmark open-source artificial general intelligence framework built on spiking neural networks and real-time learning mechanisms.

## Overview

The General Intelligence Framework (GIF) is a modular, extensible platform designed to serve as a "body" for the revolutionary **Deep Understanding (DU) Core** - a continuously learning cognitive "brain." Unlike monolithic AI models, GIF provides a plug-and-play architecture that allows researchers to attach novel sensors (Encoders) and actuators (Decoders) to a central intelligence that learns in real-time.

## Key Features

### 🧠 **Continuous Real-Time Learning (RTL)**
- Learn from constant data streams without offline retraining
- Biologically plausible plasticity mechanisms (STDP, three-factor rules)
- Online synaptic weight updates during inference

### 🔄 **Cross-Domain Generalization**
- Transfer understanding between completely different domains
- Proven ability to apply knowledge from astrophysics to medical diagnostics
- Domain-agnostic learning representations

### 📈 **System Potentiation**
- Emergent property where the framework becomes more efficient with diverse experiences
- Accelerated learning on new tasks after exposure to different domains
- Empirically validated performance improvements

### 🔌 **Plug-and-Play Architecture**
- Modular encoder/decoder interfaces
- Easy integration of new input/output modalities
- Dependency injection for maximum flexibility

### ⚡ **Neuromorphic Computing Ready**
- Event-driven computation model
- Energy-efficient spiking neural network implementation
- Hardware simulation for neuromorphic chips (Intel Loihi 2)

## Project Structure

```
gif_framework/          # Core framework implementation
├── core/              # DU Core, RTL mechanisms, memory systems
├── interfaces/        # Abstract base classes for encoders/decoders
├── training/          # Training utilities and protocols
└── utils/             # Utility functions and helpers

applications/          # Proof-of-concept applications
├── analysis/          # Analysis tools and continual learning evaluation
├── poc_exoplanet/    # Exoplanet detection demonstration
├── poc_medical/      # Medical diagnostics demonstration
└── self_generation/   # Meta-cognitive self-generation capabilities

examples/             # Interactive demonstrations
├── knowledge_augmentation_demo.py
├── meta_cognition_demo.py
├── system_potentiation_demo.py
└── vsa_deep_understanding_demo.py

data/                 # Data management
├── generated/        # Synthetic datasets (ECG, exoplanet data)
└── raw/             # External datasets

output/              # Generated visualizations and results
logs/                # System and training logs
results/             # Experimental results and analysis
data_generators/     # High-fidelity synthetic data generators
simulators/          # Neuromorphic hardware simulation
tests/              # Comprehensive test suite
docs/               # Documentation and technical reports
```

## Quick Start

### Running Examples

The framework includes several interactive demonstrations showcasing key capabilities:

```bash
# Run examples using the convenience script
python3 run_example.py meta_cognition_demo
python3 run_example.py knowledge_augmentation_demo
python3 run_example.py system_potentiation_demo
python3 run_example.py vsa_deep_understanding_demo

# Or run directly with PYTHONPATH
PYTHONPATH=. python3 examples/meta_cognition_demo.py
```

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run specific test suites
python3 -m pytest tests/test_advanced_agi_validation.py -v
python3 -m pytest tests/test_framework_structure.py -v
```

## Development Phases

The project is structured in six distinct phases:

- **Phase 0**: Foundational Architecture & Professional Tooling ✅
- **Phase 1**: Advanced Synthetic Data Generation
- **Phase 2**: Core Framework Implementation (GIF Shell & DU v1)
- **Phase 3**: Real-Time Learning & Episodic Memory
- **Phase 4**: Hardware Simulation & Proof of Concept I (Exoplanet)
- **Phase 5**: Generalization & System Potentiation (Proof of Concept II)
- **Phase 6**: Advanced Features & Community Release

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Modern package manager (`uv` recommended for best performance)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gif.git
cd gif

# Install with uv (recommended)
uv pip install -e ".[dev,snn,data]"

# Or with pip
pip install -e ".[dev,snn,data]"
```

### Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run code quality checks
ruff check .
ruff format .

# Run tests
pytest
```

## Research Foundation

This framework is based on cutting-edge research in:
- Spiking Neural Networks and Neuromorphic Computing
- Real-Time Learning and Synaptic Plasticity
- Cross-Domain Transfer Learning
- Episodic Memory Systems
- Artificial General Intelligence

## Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{gif_framework,
  title={General Intelligence Framework: A Modular AGI Platform},
  author={GIF Development Team},
  year={2025},
  url={https://github.com/your-org/gif}
}
```

## Acknowledgments

This project builds upon decades of research in computational neuroscience, neuromorphic engineering, and artificial intelligence. We thank the broader research community for their foundational contributions.
# Contributing to the General Intelligence Framework (GIF)

We welcome contributions from the research community! This document provides guidelines for contributing to the GIF project.

## Development Setup

### Prerequisites
- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- Git

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/gif.git
   cd gif
   ```

2. **Install Dependencies**
   ```bash
   # Using uv (recommended)
   uv pip install -e ".[dev,snn,data]"
   
   # Or using pip
   pip install -e ".[dev,snn,data]"
   ```

3. **Verify Installation**
   ```bash
   pytest tests/
   ruff check .
   ```

## Code Quality Standards

We maintain high code quality standards using modern Python tooling:

### Linting and Formatting
- **Ruff**: Used for both linting and formatting
- Run `ruff check .` to check for issues
- Run `ruff format .` to auto-format code
- Configuration is in `pyproject.toml`

### Testing
- **pytest**: Primary testing framework
- Aim for >90% test coverage
- Write unit tests for new functionality
- Include integration tests for complex features

### Type Hints
- Use type hints for all public APIs
- Run `mypy` for type checking
- Follow PEP 484 conventions

## Project Structure

Understanding the project architecture is crucial for contributions:

```
gif_framework/          # Core framework (stable API)
├── core/              # DU Core, RTL mechanisms, memory
├── interfaces/        # Abstract base classes
└── utils/             # Utility functions

applications/          # Proof-of-concept applications
├── poc_exoplanet/    # Exoplanet detection demo
└── poc_medical/      # Medical diagnostics demo

data_generators/       # Synthetic data generation
simulators/           # Neuromorphic hardware simulation
```

## Types of Contributions

### 🐛 Bug Reports
- Use GitHub Issues with the "bug" label
- Include minimal reproduction example
- Specify Python version and dependencies

### 🚀 Feature Requests
- Use GitHub Issues with the "enhancement" label
- Describe the use case and expected behavior
- Consider if it fits the project's scope

### 📝 Documentation
- Improve docstrings, tutorials, or examples
- Follow NumPy docstring conventions
- Include code examples where helpful

### 🧪 New Encoders/Decoders
- Implement the appropriate interface (`EncoderInterface`/`DecoderInterface`)
- Include comprehensive tests
- Provide example usage
- Document the scientific motivation

### 🧠 Core Framework Improvements
- Discuss major changes in GitHub Issues first
- Maintain backward compatibility when possible
- Include performance benchmarks for optimizations

## Submission Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   ruff check .
   ruff format .
   pytest
   ```

4. **Commit Your Changes**
   - Use clear, descriptive commit messages
   - Follow conventional commit format when possible

5. **Submit a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include test results and benchmarks

## Research Contributions

### Scientific Rigor
- Cite relevant literature in docstrings
- Include mathematical formulations where appropriate
- Validate against established benchmarks

### Reproducibility
- Include random seeds for deterministic results
- Document hyperparameters and configurations
- Provide example scripts demonstrating usage

## Community Guidelines

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Share knowledge and learn from others

## Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check existing docs first

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic publications when appropriate

Thank you for contributing to the advancement of artificial general intelligence research!

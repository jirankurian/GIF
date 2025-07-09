# GIF Framework Documentation

This directory contains comprehensive documentation for the General Intelligence Framework.

## Documentation Structure

- `architecture.md` - Detailed system architecture and design principles
- `api_reference.md` - Complete API documentation for all modules
- `tutorials/` - Step-by-step tutorials and examples
- `research/` - Research papers and theoretical foundations
- `contributing.md` - Guidelines for contributors

## Getting Started

For new users, we recommend starting with:
1. The main [README.md](../README.md) for project overview
2. `tutorials/quickstart.md` for hands-on introduction
3. `architecture.md` for understanding the system design

## Building Documentation

Documentation is built using Sphinx. To generate the full documentation:

```bash
cd docs
pip install sphinx sphinx-rtd-theme
make html
```

The generated documentation will be available in `_build/html/index.html`.

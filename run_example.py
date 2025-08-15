#!/usr/bin/env python3
"""
GIF Framework Example Runner

This script provides a convenient way to run GIF framework examples
with proper PYTHONPATH configuration.

Usage:
    python3 run_example.py <example_name>
    
Examples:
    python3 run_example.py meta_cognition_demo
    python3 run_example.py knowledge_augmentation_demo
    python3 run_example.py system_potentiation_demo
    python3 run_example.py vsa_deep_understanding_demo
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 run_example.py <example_name>")
        print("\nAvailable examples:")
        examples_dir = Path("examples")
        if examples_dir.exists():
            for example in examples_dir.glob("*.py"):
                print(f"  - {example.stem}")
        sys.exit(1)
    
    example_name = sys.argv[1]
    if not example_name.endswith('.py'):
        example_name += '.py'
    
    example_path = Path("examples") / example_name
    
    if not example_path.exists():
        print(f"Error: Example '{example_name}' not found in examples/ directory")
        sys.exit(1)
    
    # Set PYTHONPATH to include the current directory
    env = os.environ.copy()
    current_dir = Path.cwd()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{current_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(current_dir)
    
    # Run the example
    try:
        result = subprocess.run([sys.executable, str(example_path)], env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running example: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

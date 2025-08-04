"""
Vector Symbolic Architecture (VSA) Operations Module
===================================================

This module implements Vector Symbolic Architecture (also known as Hyperdimensional Computing),
providing the mathematical foundation for explicit "Deep Understanding" in the GIF framework.

VSA enables the DU_Core to form structured, semantic relationships between concepts through
hyperdimensional vector operations, transforming implicit pattern association into explicit
conceptual understanding and "connecting the dots" mechanisms.

Core Operations:
===============

1. **Hypervector Creation**: Generate high-dimensional random vectors as concept representations
2. **Binding (⊗)**: Combine two concepts into a structured relationship using circular convolution
3. **Bundling (+)**: Superpose multiple concepts into a composite representation
4. **Similarity**: Measure semantic relatedness between concepts using cosine similarity

Mathematical Foundation:
=======================

VSA operates in high-dimensional spaces (typically 10,000+ dimensions) where:
- Random vectors are nearly orthogonal (similarity ≈ 0)
- Bundled vectors are similar to their components
- Bound vectors are dissimilar to their components
- Binding is approximately invertible: (A ⊗ B) ⊗ A ≈ B

This enables powerful symbolic computation with neural-compatible representations.

Scientific Context:
==================

VSA provides a biologically plausible mechanism for symbolic reasoning in neural systems,
bridging the gap between connectionist and symbolic AI. It enables the formation of
structured knowledge representations that support compositional reasoning and concept
formation - the essence of "Deep Understanding."

Author: GIF Development Team
Phase: 7.3 - Explicit Deep Understanding via VSA
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import math
import warnings


class VSA:
    """
    Vector Symbolic Architecture implementation for hyperdimensional computing.
    
    This class provides the core operations for representing and manipulating concepts
    as high-dimensional vectors, enabling explicit symbolic reasoning within neural
    systems. The implementation follows established VSA principles while being
    optimized for PyTorch and GPU acceleration.
    
    The VSA enables "connecting the dots" by allowing the system to:
    1. Represent concepts as hypervectors
    2. Bind concepts into structured relationships
    3. Bundle related concepts into composite representations
    4. Measure semantic similarity between concepts
    
    Attributes:
        dimension (int): Dimensionality of hypervectors (typically 10,000+)
        device (torch.device): Device for tensor operations (CPU/GPU)
        dtype (torch.dtype): Data type for hypervectors (float32 by default)
    """
    
    def __init__(
        self, 
        dimension: int = 10000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None
    ):
        """
        Initialize the Vector Symbolic Architecture.
        
        Args:
            dimension (int, optional): Dimensionality of hypervectors. Higher dimensions
                                     provide better separation and more precise operations.
                                     Typical values: 1000-50000. Default: 10000.
            device (Optional[torch.device], optional): Device for tensor operations.
                                                      If None, uses default device.
            dtype (torch.dtype, optional): Data type for hypervectors. Default: float32.
            seed (Optional[int], optional): Random seed for reproducible hypervector
                                          generation. If None, uses random seed.
                                          
        Raises:
            ValueError: If dimension is not a positive integer.
            ValueError: If dimension is too small for reliable VSA operations.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")
        
        if dimension < 100:
            warnings.warn(
                f"VSA dimension {dimension} is very small. "
                f"Recommend at least 1000 for reliable operations.",
                UserWarning
            )
        
        self.dimension = dimension
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        
        # Set random seed for reproducible hypervector generation if specified
        if seed is not None:
            torch.manual_seed(seed)
        
        # Pre-compute normalization factor for efficiency
        self._norm_factor = math.sqrt(self.dimension)
    
    def create_hypervector(self, normalize: bool = True) -> torch.Tensor:
        """
        Generate a random hypervector for concept representation.
        
        Creates a high-dimensional random vector that serves as a unique identifier
        for a concept. The vector is drawn from a standard normal distribution and
        optionally normalized to unit length for consistent similarity computations.
        
        Args:
            normalize (bool, optional): Whether to normalize the hypervector to unit
                                      length. Recommended for similarity computations.
                                      Default: True.
        
        Returns:
            torch.Tensor: Random hypervector of shape [dimension] with specified dtype
                         and device. If normalized, has unit L2 norm.
        
        Example:
            vsa = VSA(dimension=1000)
            concept_vector = vsa.create_hypervector()
            print(f"Vector shape: {concept_vector.shape}")  # [1000]
            print(f"Vector norm: {torch.norm(concept_vector):.3f}")  # ≈ 1.0
        """
        # Generate random vector from standard normal distribution
        hypervector = torch.randn(
            self.dimension, 
            dtype=self.dtype, 
            device=self.device
        )
        
        if normalize:
            # Normalize to unit length for consistent similarity computations
            hypervector = F.normalize(hypervector, p=2, dim=0)
        
        return hypervector
    
    def bundle(self, hypervectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Bundle (superpose) multiple hypervectors into a composite representation.
        
        Bundling combines multiple concepts into a single hypervector that is similar
        to all its components. This operation is commutative and associative, making
        it suitable for representing sets of related concepts or accumulating knowledge.
        
        The bundled vector maintains similarity to its components while being distinct
        from any individual component, enabling hierarchical concept formation.
        
        Args:
            hypervectors (List[torch.Tensor]): List of hypervectors to bundle.
                                             Each tensor should have shape [dimension].
                                             
        Returns:
            torch.Tensor: Bundled hypervector of shape [dimension], normalized to unit
                         length. The result is similar to all input components.
                         
        Raises:
            ValueError: If hypervectors list is empty.
            ValueError: If hypervectors have inconsistent shapes or devices.
            
        Example:
            vsa = VSA(dimension=1000)
            fruit = vsa.create_hypervector()
            apple = vsa.create_hypervector()
            orange = vsa.create_hypervector()
            
            fruit_concept = vsa.bundle([apple, orange])
            
            # Bundled vector is similar to components
            assert vsa.similarity(fruit_concept, apple) > 0.4
            assert vsa.similarity(fruit_concept, orange) > 0.4
        """
        if not hypervectors:
            raise ValueError("Cannot bundle empty list of hypervectors")
        
        # Validate input tensors
        reference_shape = hypervectors[0].shape
        reference_device = hypervectors[0].device
        reference_dtype = hypervectors[0].dtype
        
        for i, hv in enumerate(hypervectors):
            if hv.shape != reference_shape:
                raise ValueError(
                    f"Hypervector {i} has shape {hv.shape}, expected {reference_shape}"
                )
            if hv.device != reference_device:
                raise ValueError(
                    f"Hypervector {i} is on device {hv.device}, expected {reference_device}"
                )
            if hv.dtype != reference_dtype:
                raise ValueError(
                    f"Hypervector {i} has dtype {hv.dtype}, expected {reference_dtype}"
                )
        
        # Element-wise addition (superposition)
        bundled = torch.stack(hypervectors, dim=0).sum(dim=0)
        
        # Normalize to unit length to maintain consistent similarity properties
        bundled = F.normalize(bundled, p=2, dim=0)
        
        return bundled
    
    def bind(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Bind two hypervectors using circular convolution to create structured relationships.
        
        Binding combines two concepts into a structured pair that is dissimilar to both
        components. This operation enables the formation of compositional representations
        like attribute-value pairs (e.g., COLOR ⊗ RED) or relational structures.
        
        The binding operation is approximately invertible: (A ⊗ B) ⊗ A ≈ B, enabling
        retrieval of associated concepts through unbinding operations.
        
        Args:
            v1 (torch.Tensor): First hypervector to bind, shape [dimension].
            v2 (torch.Tensor): Second hypervector to bind, shape [dimension].
            
        Returns:
            torch.Tensor: Bound hypervector of shape [dimension], normalized to unit
                         length. The result is dissimilar to both input vectors.
                         
        Raises:
            ValueError: If input tensors have different shapes, devices, or dtypes.
            
        Example:
            vsa = VSA(dimension=1000)
            color = vsa.create_hypervector()
            red = vsa.create_hypervector()
            
            red_color = vsa.bind(color, red)
            
            # Bound vector is dissimilar to components
            assert vsa.similarity(red_color, color) < 0.1
            assert vsa.similarity(red_color, red) < 0.1
            
            # Can retrieve red by binding with color again (approximate inverse)
            retrieved = vsa.bind(red_color, color)
            assert vsa.similarity(retrieved, red) > 0.7
        """
        # Validate input tensors
        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch: v1 {v1.shape} vs v2 {v2.shape}")
        if v1.device != v2.device:
            raise ValueError(f"Device mismatch: v1 {v1.device} vs v2 {v2.device}")
        if v1.dtype != v2.dtype:
            raise ValueError(f"Dtype mismatch: v1 {v1.dtype} vs v2 {v2.dtype}")
        
        # Perform circular convolution using FFT for efficiency
        # This is the standard binding operation in VSA
        
        # Convert to complex domain for FFT
        v1_fft = torch.fft.fft(v1)
        v2_fft = torch.fft.fft(v2)
        
        # Element-wise multiplication in frequency domain = convolution in time domain
        bound_fft = v1_fft * v2_fft
        
        # Convert back to real domain
        bound = torch.fft.ifft(bound_fft).real
        
        # Normalize to unit length to maintain consistent similarity properties
        bound = F.normalize(bound, p=2, dim=0)
        
        return bound
    
    def similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two hypervectors.
        
        Computes the cosine of the angle between two hypervectors, providing a measure
        of their semantic relatedness. Values range from -1 (opposite) to +1 (identical),
        with 0 indicating orthogonality (no relationship).
        
        For normalized hypervectors, this is equivalent to their dot product.
        
        Args:
            v1 (torch.Tensor): First hypervector, shape [dimension].
            v2 (torch.Tensor): Second hypervector, shape [dimension].
            
        Returns:
            float: Cosine similarity between the vectors, range [-1, 1].
                  Values closer to 1 indicate higher similarity.
                  
        Raises:
            ValueError: If input tensors have different shapes, devices, or dtypes.
            
        Example:
            vsa = VSA(dimension=1000)
            v1 = vsa.create_hypervector()
            v2 = vsa.create_hypervector()
            
            # Random vectors are nearly orthogonal
            sim = vsa.similarity(v1, v2)
            assert abs(sim) < 0.1  # Close to 0
            
            # Vector is identical to itself
            assert vsa.similarity(v1, v1) == 1.0
        """
        # Validate input tensors
        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch: v1 {v1.shape} vs v2 {v2.shape}")

        # Auto-fix device mismatches by moving to VSA's device
        target_device = self.device
        if v1.device != target_device:
            v1 = v1.to(target_device)
        if v2.device != target_device:
            v2 = v2.to(target_device)

        if v1.dtype != v2.dtype:
            raise ValueError(f"Dtype mismatch: v1 {v1.dtype} vs v2 {v2.dtype}")
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(v1, v2, dim=0)
        
        # Return as Python float for easier handling
        return similarity.item()
    
    def unbind(self, bound_vector: torch.Tensor, key_vector: torch.Tensor) -> torch.Tensor:
        """
        Unbind (approximately retrieve) a vector from a bound pair.
        
        Given a bound vector (A ⊗ B) and one of the original vectors (A),
        retrieves an approximation of the other vector (B). This operation
        leverages the approximate invertibility of circular convolution.
        
        Args:
            bound_vector (torch.Tensor): The bound vector (A ⊗ B), shape [dimension].
            key_vector (torch.Tensor): One of the original vectors (A), shape [dimension].
            
        Returns:
            torch.Tensor: Approximation of the other original vector (B), shape [dimension].
                         
        Note:
            This is an approximate operation. The retrieved vector will be similar
            to but not identical to the original vector due to noise in the binding process.
        """
        # Unbinding is the same operation as binding due to circular convolution properties
        return self.bind(bound_vector, key_vector)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get VSA configuration and statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing VSA configuration and operational statistics.
        """
        return {
            'dimension': self.dimension,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'norm_factor': self._norm_factor,
            'memory_per_vector_mb': (self.dimension * 4) / (1024 * 1024),  # Assuming float32
        }

"""
Base Interfaces for the General Intelligence Framework (GIF)
===========================================================

This module defines the foundational software contracts that establish the plug-and-play
architecture of the GIF framework. These abstract base classes (ABCs) serve as the
"blueprints" that all modular components must follow, ensuring true modularity and
extensibility.

The interfaces defined here are the cornerstone of the GIF's architectural philosophy:
- Separation of concerns between encoding, processing, and decoding
- Dependency injection enabling flexible component composition
- Contract-based design ensuring interoperability
- Modular "plug-and-play" ecosystem for different domains and tasks

Key Components:
- SpikeTrain: Type alias for standardized spike train representation
- Action: Type alias for flexible decoder output types
- EncoderInterface: Contract for all input encoders
- DecoderInterface: Contract for all output decoders

This design directly implements the Object-Oriented Programming (OOP) and Dependency
Injection (DI) principles central to creating a truly modular and scalable framework.

Author: GIF Development Team
Phase: 2.1 - Core Framework Implementation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


# =============================================================================
# Type Aliases for Framework Standardization
# =============================================================================

# SpikeTrain represents the standardized format for neural spike data throughout
# the GIF framework. Using PyTorch tensors enables efficient computation and
# seamless integration with the SNN-based DU Core.
SpikeTrain = torch.Tensor

# Action represents the final output of a decoder. It is intentionally flexible
# (Any type) because decoder outputs can vary widely:
# - Classification: string or integer labels
# - Regression: numerical values or arrays
# - Complex Actions: dictionaries with multiple parameters
# - Robot Commands: structured command objects
# This flexibility is essential for the framework's domain-agnostic design.
Action = Any


# =============================================================================
# Encoder Interface - Input Processing Contract
# =============================================================================

class EncoderInterface(ABC):
    """
    Abstract base class defining the contract for all input encoders in the GIF framework.
    
    Encoders are responsible for transforming raw input data from any domain (e.g., 
    astronomical light curves, medical ECG signals, images, text, sensor data) into 
    the framework's standardized SpikeTrain format suitable for processing by the 
    DU Core.
    
    This interface ensures that the central GIF orchestrator can work with any encoder
    implementation without knowing its internal details, as long as it honors this
    contract. This is the foundation of the plug-and-play architecture.
    
    Key Design Principles:
    - Domain Agnostic: Can handle any type of input data
    - Standardized Output: Always produces SpikeTrain format
    - Configurable: Supports parameter retrieval for reproducibility
    - Adaptive: Includes calibration hook for Real-Time Learning (RTL)
    
    Example Usage:
        class MyCustomEncoder(EncoderInterface):
            def encode(self, raw_data):
                # Convert raw_data to spike train
                return spike_train
            
            def get_config(self):
                return {"param1": value1, "param2": value2}
            
            def calibrate(self, sample_data):
                # Adjust parameters based on sample_data
                pass
    """
    
    @abstractmethod
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert raw input data into a standardized spike train format.
        
        This is the core method of any encoder. It takes raw data in any format
        (NumPy arrays, Polars DataFrames, strings, images, etc.) and transforms
        it into the framework's standardized SpikeTrain representation.
        
        The encoding process typically involves:
        1. Data preprocessing and normalization
        2. Feature extraction or transformation
        3. Conversion to temporal spike patterns
        4. Formatting as PyTorch tensor
        
        Args:
            raw_data (Any): Input data in any format. The specific type and
                          structure depend on the encoder implementation and
                          the domain it serves (e.g., time series, images, text).
        
        Returns:
            SpikeTrain: A PyTorch tensor representing the encoded spike train.
                       The tensor shape and temporal structure depend on the
                       specific encoding strategy but must be compatible with
                       the DU Core's input requirements.
        
        Raises:
            ValueError: If the input data cannot be processed or is invalid.
            TypeError: If the input data type is not supported by this encoder.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the current configuration parameters of this encoder instance.
        
        This method is essential for reproducibility, logging, and debugging.
        It should return all configurable parameters that affect the encoding
        process, allowing for exact reconstruction of the encoder's behavior.
        
        Returns:
            Dict[str, Any]: A dictionary containing all configuration parameters.
                          Keys should be descriptive parameter names, and values
                          should be the current parameter values. This should
                          include encoding method, thresholds, normalization
                          parameters, and any other settings that affect output.
        
        Example:
            {
                "encoding_method": "rate_coding",
                "time_window": 100,
                "threshold": 0.5,
                "normalization": "min_max",
                "spike_rate": 50.0
            }
        """
        pass
    
    @abstractmethod
    def calibrate(self, sample_data: Any) -> None:
        """
        Calibrate encoder parameters based on a sample of new data.
        
        This method provides the hook for Real-Time Learning (RTL) driven 
        auto-calibration as described in the GIF research papers. The GIF 
        orchestrator can call this method to allow the encoder to dynamically 
        adjust its parameters when encountering new, unseen data distributions.
        
        The calibration process might involve:
        - Updating normalization parameters based on data statistics
        - Adjusting encoding thresholds for optimal spike generation
        - Modifying temporal parameters based on signal characteristics
        - Learning domain-specific feature representations
        
        Args:
            sample_data (Any): A representative sample of new data that the
                             encoder should adapt to. The format should match
                             the expected input format for the encode() method.
        
        Returns:
            None: This method modifies the encoder's internal state but does
                 not return a value. The effects should be visible in subsequent
                 calls to encode() and get_config().
        
        Note:
            This method is called by the GIF orchestrator during adaptive
            learning phases. Implementations should be efficient and robust,
            as they may be called frequently during online learning scenarios.
        """
        pass


# =============================================================================
# Decoder Interface - Output Processing Contract
# =============================================================================

class DecoderInterface(ABC):
    """
    Abstract base class defining the contract for all output decoders in the GIF framework.
    
    Decoders are responsible for transforming spike trains from the DU Core into 
    meaningful, domain-specific outputs. These outputs can be classifications, 
    predictions, control signals, or any other form of actionable result.
    
    This interface ensures that the central GIF orchestrator can work with any decoder
    implementation without knowing its internal details, enabling the same DU Core
    to drive completely different types of applications through different decoders.
    
    Key Design Principles:
    - Domain Specific: Produces outputs appropriate for the target application
    - Flexible Output: Can generate any type of action or result
    - Configurable: Supports parameter retrieval for reproducibility
    - Standardized Input: Always accepts SpikeTrain format from DU Core
    
    Example Usage:
        class MyCustomDecoder(DecoderInterface):
            def decode(self, spike_train):
                # Convert spike train to meaningful output
                return action
            
            def get_config(self):
                return {"param1": value1, "param2": value2}
    """
    
    @abstractmethod
    def decode(self, spike_train: SpikeTrain) -> Action:
        """
        Convert a spike train from the DU Core into a meaningful output action.
        
        This is the core method of any decoder. It takes the processed spike train
        from the DU Core and translates it into a final, domain-specific output
        that can be used by the target application.
        
        The decoding process typically involves:
        1. Spike pattern analysis and feature extraction
        2. Temporal integration and pattern recognition
        3. Classification, regression, or decision making
        4. Formatting as appropriate output type
        
        Args:
            spike_train (SpikeTrain): A PyTorch tensor containing the processed
                                    spike train from the DU Core. The tensor
                                    represents temporal neural activity patterns
                                    that encode the DU Core's "understanding"
                                    of the input.
        
        Returns:
            Action: The decoded output in a format appropriate for the target
                   application. This could be:
                   - Classification labels (string, int)
                   - Regression values (float, array)
                   - Control commands (dict, object)
                   - Complex structured outputs (custom objects)
        
        Raises:
            ValueError: If the spike train cannot be decoded or is malformed.
            RuntimeError: If the decoder is not properly configured.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the current configuration parameters of this decoder instance.
        
        This method serves the same purpose as the encoder's get_config method:
        ensuring reproducibility, enabling logging, and supporting debugging.
        It should return all parameters that affect the decoding process.
        
        Returns:
            Dict[str, Any]: A dictionary containing all configuration parameters.
                          Keys should be descriptive parameter names, and values
                          should be the current parameter values. This should
                          include decoding method, thresholds, output formatting
                          parameters, and any other settings that affect output.
        
        Example:
            {
                "decoding_method": "spike_count",
                "integration_window": 50,
                "decision_threshold": 0.7,
                "output_classes": ["class_a", "class_b", "class_c"],
                "smoothing_factor": 0.1
            }
        """
        pass

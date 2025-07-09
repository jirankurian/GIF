"""
Central Orchestrator for the General Intelligence Framework (GIF)
================================================================

This module implements the central nervous system of the GIF framework - the `GIF` class
that acts as the primary orchestrator for all modular components. The orchestrator manages
the lifecycle of encoders, decoders, and the DU Core, directing the flow of information
through the complete cognitive cycle.

The GIF class embodies two critical software engineering patterns:

1. **Orchestrator Pattern**: The GIF class coordinates the sequence of operations
   (encode → process → decode) without containing domain-specific logic itself.
   It knows the "what" and "when" but not the "how" of each operation.

2. **Dependency Injection (DI)**: The GIF class receives its "brain" (DU Core) as
   a constructor parameter rather than creating it internally. This makes the
   framework highly flexible and testable - the same GIF body can work with
   different brains without any code changes.

Key Architectural Principles:
- Separation of Concerns: Coordination logic separate from implementation logic
- Interface-Based Design: Uses abstract interfaces to ensure modularity
- Plug-and-Play Architecture: Components can be swapped without code changes
- Contract Enforcement: Type hints ensure only valid components can be attached
- Error Handling: Clear validation and descriptive error messages

This design directly implements the GIF research vision of separating the immutable
"brain" (DU Core) from the attachable "body parts" (encoders/decoders), enabling
the same intelligence to drive completely different applications.

Author: GIF Development Team
Phase: 2.2 - Core Framework Implementation
"""

from typing import Any, Optional, Dict, Union
import numpy as np
import logging
from gif_framework.interfaces.base_interfaces import (
    EncoderInterface,
    DecoderInterface,
    SpikeTrain,
    Action
)

# Import module library for meta-cognitive routing
try:
    from gif_framework.module_library import ModuleLibrary, ModuleMetadata
except ImportError:
    ModuleLibrary = None
    ModuleMetadata = None

# Import DU_Core_V1 for type hinting (will be implemented in Task 2.3)
# Python allows forward references for type hints even when the class doesn't exist yet
if False:  # TYPE_CHECKING equivalent - prevents runtime import errors
    from gif_framework.core.du_core import DU_Core_V1


class MetaController:
    """
    Meta-cognitive controller for intelligent data analysis and module selection.

    The MetaController represents the "self-awareness" component of the GIF framework,
    capable of analyzing raw data to understand its characteristics and generating
    metadata that can be used for intelligent module selection. This meta-cognitive
    capability enables the system to reason about its own tools and make optimal
    decisions about which modules to use for specific tasks.

    The controller implements sophisticated signal analysis algorithms to automatically
    characterize data across multiple dimensions:
    - Data modality (timeseries, image, text, etc.)
    - Signal characteristics (periodic, burst, continuous)
    - Domain inference (medical, astronomical, general)
    - Format detection (array, dataframe, tensor)
    - Complexity assessment (simple, moderate, complex)

    This analysis enables the ModuleLibrary to make intelligent module selections
    without human intervention, representing a significant step toward AGI.

    Example:
        controller = MetaController()

        # Analyze ECG data
        ecg_metadata = controller.analyze_data(ecg_signal)
        # Returns: {'data_modality': 'timeseries', 'signal_type': 'periodic',
        #           'domain': 'medical', 'input_format': 'dataframe'}

        # Analyze light curve data
        lc_metadata = controller.analyze_data(light_curve)
        # Returns: {'data_modality': 'timeseries', 'signal_type': 'periodic',
        #           'domain': 'astronomy', 'input_format': 'array'}
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the meta-controller."""
        self.logger = logger or logging.getLogger(__name__)

    def analyze_data(self, raw_data: Any) -> Dict[str, Any]:
        """
        Analyze raw data to determine its characteristics and generate metadata.

        This method implements the core meta-cognitive capability by examining
        the structure, content, and statistical properties of input data to
        automatically determine its characteristics. This analysis enables
        intelligent module selection without human intervention.

        The analysis considers multiple dimensions:
        1. Data format and structure
        2. Signal characteristics and patterns
        3. Domain-specific indicators
        4. Complexity and processing requirements

        Args:
            raw_data: Input data in any format (array, dataframe, tensor, etc.)

        Returns:
            Dict[str, Any]: Metadata describing the data characteristics

        Example:
            # Analyze periodic timeseries data
            metadata = controller.analyze_data(ecg_signal)
            # Returns: {
            #     'data_modality': 'timeseries',
            #     'signal_type': 'periodic',
            #     'domain': 'medical',
            #     'input_format': 'dataframe',
            #     'complexity': 'moderate',
            #     'length': 5000,
            #     'channels': 3
            # }
        """
        try:
            # Step 1: Determine data format
            input_format = self._detect_input_format(raw_data)

            # Step 2: Extract numerical data for analysis
            numerical_data = self._extract_numerical_data(raw_data)

            # Step 3: Determine data modality
            data_modality = self._detect_data_modality(numerical_data)

            # Step 4: Analyze signal characteristics (for timeseries)
            signal_type = 'unknown'
            if data_modality == 'timeseries':
                signal_type = self._analyze_signal_characteristics(numerical_data)

            # Step 5: Infer domain
            domain = self._infer_domain(numerical_data, signal_type)

            # Step 6: Assess complexity
            complexity = self._assess_complexity(numerical_data)

            # Step 7: Extract structural information
            structure_info = self._extract_structure_info(numerical_data)

            # Compile metadata
            metadata = {
                'data_modality': data_modality,
                'signal_type': signal_type,
                'domain': domain,
                'input_format': input_format,
                'complexity': complexity,
                **structure_info
            }

            self.logger.debug(f"Data analysis complete: {metadata}")
            return metadata

        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}")
            # Return minimal fallback metadata
            return {
                'data_modality': 'unknown',
                'signal_type': 'unknown',
                'domain': 'general',
                'input_format': 'unknown',
                'complexity': 'unknown'
            }

    def _detect_input_format(self, raw_data: Any) -> str:
        """Detect the format of input data."""
        try:
            import pandas as pd
            if isinstance(raw_data, pd.DataFrame):
                return 'dataframe'
        except ImportError:
            pass

        try:
            import torch
            if isinstance(raw_data, torch.Tensor):
                return 'tensor'
        except ImportError:
            pass

        if isinstance(raw_data, np.ndarray):
            return 'array'
        elif isinstance(raw_data, (list, tuple)):
            return 'list'
        elif isinstance(raw_data, dict):
            return 'dict'
        else:
            return 'unknown'

    def _extract_numerical_data(self, raw_data: Any) -> np.ndarray:
        """Extract numerical data for analysis."""
        try:
            import pandas as pd
            if isinstance(raw_data, pd.DataFrame):
                # Extract numerical columns
                numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    return raw_data[numeric_cols].values
                else:
                    return np.array([])
        except ImportError:
            pass

        try:
            import torch
            if isinstance(raw_data, torch.Tensor):
                return raw_data.detach().cpu().numpy()
        except ImportError:
            pass

        if isinstance(raw_data, np.ndarray):
            return raw_data
        elif isinstance(raw_data, (list, tuple)):
            return np.array(raw_data)
        else:
            return np.array([])

    def _detect_data_modality(self, data: np.ndarray) -> str:
        """Detect the modality of the data."""
        if data.size == 0:
            return 'unknown'

        # Check dimensions
        if data.ndim == 1 or (data.ndim == 2 and min(data.shape) <= 10):
            return 'timeseries'
        elif data.ndim == 2 and min(data.shape) > 10:
            return 'image'
        elif data.ndim == 3:
            return 'image'  # Could be RGB image or video frame
        else:
            return 'signal'  # Generic signal data

    def _analyze_signal_characteristics(self, data: np.ndarray) -> str:
        """Analyze signal characteristics for timeseries data."""
        if data.size == 0:
            return 'unknown'

        try:
            # Flatten if multi-channel
            if data.ndim > 1:
                signal = data.flatten()
            else:
                signal = data

            # Remove DC component
            signal = signal - np.mean(signal)

            # Calculate power spectrum
            if len(signal) > 10:
                fft = np.fft.fft(signal)
                power_spectrum = np.abs(fft) ** 2

                # Find dominant frequency
                freqs = np.fft.fftfreq(len(signal))
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1

                # Check for periodicity
                max_power = power_spectrum[dominant_freq_idx]
                total_power = np.sum(power_spectrum[1:len(power_spectrum)//2])

                if max_power / total_power > 0.3:  # Strong periodic component
                    return 'periodic'
                elif max_power / total_power > 0.1:  # Weak periodic component
                    return 'quasi_periodic'
                else:
                    # Check for burst patterns
                    signal_std = np.std(signal)
                    high_activity = np.sum(np.abs(signal) > 2 * signal_std)

                    if high_activity / len(signal) < 0.1:
                        return 'burst'
                    else:
                        return 'continuous'
            else:
                return 'discrete'

        except Exception:
            return 'unknown'

    def _infer_domain(self, data: np.ndarray, signal_type: str) -> str:
        """Infer the application domain based on data characteristics."""
        if data.size == 0:
            return 'general'

        try:
            # Simple heuristics for domain inference
            if data.ndim <= 2 and signal_type == 'periodic':
                # Check for medical-like characteristics
                if data.ndim == 2 and data.shape[1] <= 12:  # Multi-lead ECG-like
                    return 'medical'
                elif len(data) > 1000:  # Long timeseries could be astronomical
                    return 'astronomy'

            # Check value ranges for domain hints
            if np.all(data >= 0) and np.max(data) < 2:
                # Normalized data, could be light curves
                return 'astronomy'
            elif np.any(data < 0) and signal_type == 'periodic':
                # Bipolar signals, likely medical
                return 'medical'

            return 'general'

        except Exception:
            return 'general'

    def _assess_complexity(self, data: np.ndarray) -> str:
        """Assess the complexity of the data for processing requirements."""
        if data.size == 0:
            return 'unknown'

        try:
            # Simple complexity assessment based on size and dimensionality
            total_elements = data.size

            if total_elements < 1000:
                return 'simple'
            elif total_elements < 100000:
                return 'moderate'
            else:
                return 'complex'

        except Exception:
            return 'unknown'

    def _extract_structure_info(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract structural information about the data."""
        if data.size == 0:
            return {}

        try:
            info = {
                'shape': list(data.shape),
                'total_elements': int(data.size),
                'dtype': str(data.dtype)
            }

            # Add dimension-specific info
            if data.ndim == 1:
                info['length'] = len(data)
            elif data.ndim == 2:
                info['length'] = data.shape[0]
                info['channels'] = data.shape[1]
            elif data.ndim == 3:
                info['height'] = data.shape[0]
                info['width'] = data.shape[1]
                info['channels'] = data.shape[2]

            return info

        except Exception:
            return {}


class GIF:
    """
    The central orchestrator class for the General Intelligence Framework.
    
    The GIF class serves as the "body" of the artificial intelligence system,
    managing the flow of information between modular components while remaining
    completely agnostic to the specific domains or tasks being performed.
    
    This class implements the core operational cycle of the GIF framework:
    1. Raw data enters through an attached Encoder
    2. Encoded spike trains are processed by the injected DU Core ("brain")
    3. Processed spike trains are converted to actions by an attached Decoder
    4. Final actions are returned to the calling application
    
    The orchestrator uses Dependency Injection for the DU Core and interface-based
    attachment for encoders and decoders, ensuring maximum flexibility and testability.
    
    Key Design Features:
    - **Domain Agnostic**: Works with any encoder/decoder that implements the interfaces
    - **Brain Agnostic**: Can work with any DU Core implementation via dependency injection
    - **Plug-and-Play**: Components can be attached, detached, and swapped at runtime
    - **Type Safe**: Uses interface type hints to enforce architectural contracts
    - **Error Resilient**: Comprehensive validation and clear error messages
    
    Example Usage:
        # Create a DU Core instance (the "brain")
        du_core = DU_Core_V1(input_size=100, hidden_sizes=[64, 32], output_size=10)
        
        # Create the GIF orchestrator with dependency injection
        gif = GIF(du_core)
        
        # Attach domain-specific components
        gif.attach_encoder(MyCustomEncoder())
        gif.attach_decoder(MyCustomDecoder())
        
        # Process data through the complete cognitive cycle
        result = gif.process_single_input(raw_data)
    
    Attributes:
        _du_core: The injected Deep Understanding core (the "brain")
        _encoder: The currently attached encoder (None if not attached)
        _decoder: The currently attached decoder (None if not attached)
    """
    
    def __init__(
        self,
        du_core: 'DU_Core_V1',
        module_library: Optional['ModuleLibrary'] = None
    ) -> None:
        """
        Initialize the GIF orchestrator with a DU Core instance and optional module library.

        This constructor implements the Dependency Injection pattern - the GIF
        orchestrator receives its "brain" as a parameter rather than creating
        it internally. The optional module library enables meta-cognitive routing
        capabilities for intelligent module selection.

        Args:
            du_core (DU_Core_V1): An instance of the Deep Understanding core
                                 that will serve as the "brain" of this GIF
                                 instance. The DU core must implement the
                                 expected interface with a process() method.
            module_library (Optional[ModuleLibrary]): Optional module library
                                                     for meta-cognitive routing.
                                                     If provided, enables automatic
                                                     module selection based on data
                                                     characteristics.

        Raises:
            TypeError: If du_core is None or not a valid DU core instance.

        Note:
            With module_library: Use auto_configure_for_data() for intelligent setup
            Without module_library: Use attach_encoder/decoder() for manual setup
        """
        if du_core is None:
            raise TypeError("DU Core cannot be None. Please provide a valid DU_Core_V1 instance.")
        
        # Store the injected brain
        self._du_core = du_core

        # Store module library for meta-cognitive routing
        self._module_library = module_library

        # Initialize meta-controller if module library is available
        if module_library and ModuleLibrary is not None:
            self._meta_controller = MetaController()
        else:
            self._meta_controller = None

        # Initialize attachment points for modular components
        # Using Optional type hints makes the interface self-documenting
        self._encoder: Optional[EncoderInterface] = None
        self._decoder: Optional[DecoderInterface] = None

        # Track whether modules were auto-selected
        self._auto_configured = False
    
    def attach_encoder(self, encoder: EncoderInterface) -> None:
        """
        Attach an encoder to the GIF orchestrator.
        
        This method implements the plug-and-play architecture by allowing
        encoders to be attached at runtime. The encoder type hint ensures
        that only objects implementing the EncoderInterface contract can
        be attached, enforcing the architectural design.
        
        Args:
            encoder (EncoderInterface): An encoder instance that implements
                                      the EncoderInterface contract. The encoder
                                      will be responsible for converting raw
                                      input data into spike trains.
        
        Raises:
            TypeError: If encoder is None or doesn't implement EncoderInterface.
        
        Example:
            gif.attach_encoder(ExoplanetEncoder())
            gif.attach_encoder(ECGEncoder())  # Can swap encoders at runtime
        
        Note:
            Attaching a new encoder will replace any previously attached encoder.
            This enables dynamic reconfiguration for different data types.
        """
        if encoder is None:
            raise TypeError("Encoder cannot be None. Please provide a valid EncoderInterface implementation.")
        
        if not isinstance(encoder, EncoderInterface):
            raise TypeError(f"Encoder must implement EncoderInterface. Got {type(encoder).__name__}")
        
        self._encoder = encoder
    
    def attach_decoder(self, decoder: DecoderInterface) -> None:
        """
        Attach a decoder to the GIF orchestrator.
        
        This method implements the plug-and-play architecture by allowing
        decoders to be attached at runtime. The decoder type hint ensures
        that only objects implementing the DecoderInterface contract can
        be attached, enforcing the architectural design.
        
        Args:
            decoder (DecoderInterface): A decoder instance that implements
                                      the DecoderInterface contract. The decoder
                                      will be responsible for converting processed
                                      spike trains into meaningful actions.
        
        Raises:
            TypeError: If decoder is None or doesn't implement DecoderInterface.
        
        Example:
            gif.attach_decoder(ClassificationDecoder())
            gif.attach_decoder(RegressionDecoder())  # Can swap decoders at runtime
        
        Note:
            Attaching a new decoder will replace any previously attached decoder.
            This enables dynamic reconfiguration for different output types.
        """
        if decoder is None:
            raise TypeError("Decoder cannot be None. Please provide a valid DecoderInterface implementation.")
        
        if not isinstance(decoder, DecoderInterface):
            raise TypeError(f"Decoder must implement DecoderInterface. Got {type(decoder).__name__}")
        
        self._decoder = decoder

    def auto_configure_for_data(
        self,
        raw_data: Any,
        performance_preference: str = 'balanced'
    ) -> bool:
        """
        Automatically configure encoder and decoder based on data characteristics.

        This method implements the meta-cognitive routing capability by analyzing
        the input data and automatically selecting the most appropriate encoder
        and decoder from the module library. This represents a significant step
        toward AGI by enabling the system to reason about its own tools.

        The method uses the MetaController to analyze data characteristics and
        the ModuleLibrary to find the best matching modules based on multiple
        compatibility factors.

        Args:
            raw_data: Input data to analyze for module selection
            performance_preference: 'accuracy', 'speed', 'balanced', 'memory'

        Returns:
            bool: True if modules were successfully selected, False otherwise

        Raises:
            RuntimeError: If no module library is available

        Example:
            # Automatic configuration for ECG data
            gif = GIF(du_core, module_library)
            success = gif.auto_configure_for_data(ecg_signal)
            if success:
                result = gif.process_single_input(ecg_signal)
        """
        if self._module_library is None or self._meta_controller is None:
            raise RuntimeError(
                "Module library not available. Please provide a ModuleLibrary "
                "instance during GIF initialization to enable meta-cognitive routing."
            )

        try:
            # Step 1: Analyze data characteristics
            data_metadata = self._meta_controller.analyze_data(raw_data)

            # Step 2: Find best matching modules
            encoder, decoder = self._module_library.find_best_match(
                data_metadata, performance_preference
            )

            # Step 3: Configure the system
            if encoder is not None and decoder is not None:
                self._encoder = encoder
                self._decoder = decoder
                self._auto_configured = True

                # Log successful configuration
                encoder_name = type(encoder).__name__
                decoder_name = type(decoder).__name__

                print(f"✅ Auto-configured: {encoder_name} + {decoder_name}")
                print(f"   Data: {data_metadata.get('domain', 'unknown')}/{data_metadata.get('data_modality', 'unknown')}")
                print(f"   Signal: {data_metadata.get('signal_type', 'unknown')}")

                return True
            else:
                print(f"❌ No suitable modules found for data characteristics:")
                print(f"   {data_metadata}")
                return False

        except Exception as e:
            print(f"❌ Auto-configuration failed: {e}")
            return False

    def process_single_input(self, raw_data: Any) -> Action:
        """
        Execute a complete cognitive cycle on a single input.
        
        This is the core operational method of the GIF framework. It orchestrates
        the complete flow of information through the system:
        1. Validation: Ensures all required components are attached
        2. Encoding: Converts raw data to standardized spike trains
        3. Processing: Passes spike trains through the DU Core for "understanding"
        4. Decoding: Converts processed spikes to meaningful actions
        5. Return: Provides the final result to the calling application
        
        The method implements comprehensive error checking to ensure the system
        is properly configured before attempting to process data.
        
        Args:
            raw_data (Any): Input data in any format supported by the attached
                          encoder. The specific format depends on the encoder
                          implementation (e.g., time series, images, text).
        
        Returns:
            Action: The final output from the decoder, which could be:
                   - Classification results (strings, integers)
                   - Regression values (floats, arrays)
                   - Control commands (dictionaries, objects)
                   - Any other domain-specific output type
        
        Raises:
            RuntimeError: If encoder or decoder is not attached.
            ValueError: If the input data cannot be processed by the encoder.
            Exception: Any exception raised by the encoder, DU core, or decoder
                      will be propagated with additional context.
        
        Example:
            # Process astronomical data
            result = gif.process_single_input(light_curve_data)
            
            # Process medical data
            diagnosis = gif.process_single_input(ecg_signal)
        
        Note:
            This method represents a single "thought" of the artificial intelligence.
            For continuous processing, call this method repeatedly with new data.
        """
        # Step 1: Validate that all required components are attached
        if self._encoder is None:
            raise RuntimeError(
                "No encoder attached. Please attach an encoder using attach_encoder() "
                "before processing input data."
            )
        
        if self._decoder is None:
            raise RuntimeError(
                "No decoder attached. Please attach a decoder using attach_decoder() "
                "before processing input data."
            )
        
        try:
            # Step 2: Encode - Convert raw data to spike trains
            spike_train: SpikeTrain = self._encoder.encode(raw_data)
            
            # Step 3: Process - Pass through the DU Core for "understanding"
            # Note: The DU Core's process method will be implemented in Task 2.3
            processed_spikes: SpikeTrain = self._du_core.process(spike_train)
            
            # Step 4: Decode - Convert processed spikes to meaningful actions
            action: Action = self._decoder.decode(processed_spikes)
            
            # Step 5: Return the final result
            return action
            
        except Exception as e:
            # Provide additional context for debugging while preserving the original exception
            raise type(e)(
                f"Error during cognitive cycle processing: {str(e)}. "
                f"Encoder: {type(self._encoder).__name__}, "
                f"DU Core: {type(self._du_core).__name__}, "
                f"Decoder: {type(self._decoder).__name__}"
            ) from e
    
    def get_system_config(self) -> dict[str, Any]:
        """
        Get the complete configuration of the GIF system.
        
        This method provides a comprehensive view of the current system state,
        including the configuration of all attached components. This is essential
        for reproducibility, debugging, and system monitoring.
        
        Returns:
            dict[str, Any]: A dictionary containing:
                - du_core_config: Configuration of the DU Core
                - encoder_config: Configuration of attached encoder (if any)
                - decoder_config: Configuration of attached decoder (if any)
                - system_status: Overall system readiness status
        
        Example:
            config = gif.get_system_config()
            print(f"System ready: {config['system_status']['ready']}")
        """
        config = {
            "du_core_config": getattr(self._du_core, 'get_config', lambda: {})(),
            "encoder_config": self._encoder.get_config() if self._encoder else None,
            "decoder_config": self._decoder.get_config() if self._decoder else None,
            "system_status": {
                "ready": self._encoder is not None and self._decoder is not None,
                "encoder_attached": self._encoder is not None,
                "decoder_attached": self._decoder is not None,
                "encoder_type": type(self._encoder).__name__ if self._encoder else None,
                "decoder_type": type(self._decoder).__name__ if self._decoder else None,
                "du_core_type": type(self._du_core).__name__,
                "meta_cognitive_enabled": self._module_library is not None,
                "auto_configured": self._auto_configured
            },
            "meta_cognitive_config": {
                "module_library_available": self._module_library is not None,
                "meta_controller_available": self._meta_controller is not None,
                "library_stats": self._module_library.get_library_stats() if self._module_library else None
            }
        }
        return config

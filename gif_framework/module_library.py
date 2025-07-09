"""
Module Library - Meta-Cognitive Module Registry
==============================================

This module implements the Module Library system that enables meta-cognitive routing
in the GIF framework. The library acts as an intelligent registry that stores
encoder and decoder modules along with their capability metadata, allowing the
system to automatically select the most appropriate modules for any given data type.

The Module Library represents a significant advancement toward true AGI by enabling
the system to reason about its own capabilities and make intelligent decisions
about which tools to use for specific tasks. This meta-cognitive ability is
inspired by frameworks like SymRAG and represents "thinking about thinking."

Key Features:
- **Intelligent Module Registry**: Store modules with rich capability metadata
- **Automatic Module Selection**: Find best-matching modules for data characteristics
- **Capability Reasoning**: Understand module strengths and limitations
- **Performance Optimization**: Select modules based on accuracy/latency trade-offs
- **Extensible Architecture**: Easy addition of new modules and metadata fields

Technical Innovation:
The library uses a sophisticated matching algorithm that considers multiple
dimensions of compatibility:
- Data modality (timeseries, image, text, etc.)
- Signal characteristics (periodic, burst, continuous)
- Domain specificity (medical, astronomical, general)
- Performance requirements (latency, accuracy, memory)
- Input/output format compatibility

This enables the GIF framework to automatically configure itself for optimal
performance on any task without human intervention.

Author: GIF Development Team
Phase: 6.3 - Advanced Features Implementation
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import interfaces for type checking
from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface


@dataclass
class ModuleMetadata:
    """
    Structured metadata describing a module's capabilities and characteristics.
    
    This dataclass provides a standardized way to describe module capabilities,
    enabling intelligent matching and selection. The metadata covers multiple
    dimensions of module characteristics to support sophisticated reasoning.
    
    Attributes:
        module_type (str): Type of module ('encoder' or 'decoder')
        data_modality (str): Primary data type ('timeseries', 'image', 'text', 'signal')
        signal_type (str): Signal characteristics ('periodic', 'burst', 'continuous', 'discrete')
        domain (str): Application domain ('medical', 'astronomy', 'general', 'robotics')
        input_format (str): Expected input format ('dataframe', 'array', 'tensor', 'dict')
        output_format (str): Produced output format ('tensor', 'classification', 'regression')
        capabilities (List[str]): Specific capabilities and features
        performance_profile (Dict[str, str]): Performance characteristics
        compatibility_score (float): General compatibility rating (0.0-1.0)
        priority (int): Selection priority (higher = preferred)
        
    Example:
        metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='periodic',
            domain='medical',
            input_format='dataframe',
            capabilities=['r_peak_detection', 'heart_rate_analysis'],
            performance_profile={'latency': 'low', 'accuracy': 'high'}
        )
    """
    module_type: str  # 'encoder' or 'decoder'
    data_modality: str  # 'timeseries', 'image', 'text', 'signal'
    signal_type: str  # 'periodic', 'burst', 'continuous', 'discrete'
    domain: str  # 'medical', 'astronomy', 'general', 'robotics'
    input_format: str  # 'dataframe', 'array', 'tensor', 'dict'
    output_format: str = 'tensor'  # 'tensor', 'classification', 'regression'
    capabilities: List[str] = field(default_factory=list)
    performance_profile: Dict[str, str] = field(default_factory=dict)
    compatibility_score: float = 1.0  # 0.0-1.0 general compatibility
    priority: int = 1  # Higher = preferred
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate metadata fields after initialization."""
        valid_types = ['encoder', 'decoder']
        if self.module_type not in valid_types:
            raise ValueError(f"module_type must be one of {valid_types}")
            
        if not (0.0 <= self.compatibility_score <= 1.0):
            raise ValueError("compatibility_score must be between 0.0 and 1.0")


@dataclass
class ModuleEntry:
    """
    Registry entry combining a module instance with its metadata.
    
    This class represents a complete module registration, including both
    the functional module and its descriptive metadata for intelligent
    selection and reasoning.
    
    Attributes:
        module: The actual encoder or decoder instance
        metadata: Structured metadata describing the module
        registration_id: Unique identifier for this registration
    """
    module: Union[EncoderInterface, DecoderInterface]
    metadata: ModuleMetadata
    registration_id: str


class ModuleLibrary:
    """
    Intelligent module registry enabling meta-cognitive routing.
    
    The ModuleLibrary serves as the "self-awareness" component of the GIF framework,
    maintaining a registry of available modules along with rich metadata about their
    capabilities. This enables the system to reason about its own tools and make
    intelligent decisions about which modules to use for specific tasks.
    
    The library implements sophisticated matching algorithms that consider multiple
    dimensions of compatibility, including data characteristics, performance
    requirements, and domain specificity. This meta-cognitive capability represents
    a significant step toward true AGI.
    
    Key Features:
    - **Module Registration**: Store modules with comprehensive metadata
    - **Intelligent Matching**: Find optimal modules for data characteristics
    - **Multi-criteria Selection**: Consider accuracy, latency, and compatibility
    - **Capability Reasoning**: Understand module strengths and limitations
    - **Extensible Design**: Easy addition of new modules and metadata fields
    
    Example:
        # Create library and register modules
        library = ModuleLibrary()
        
        # Register ECG encoder
        ecg_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='periodic',
            domain='medical',
            capabilities=['r_peak_detection', 'heart_rate_analysis']
        )
        library.register_module(ecg_encoder, ecg_metadata)
        
        # Automatic module selection
        encoder, decoder = library.find_best_match(data_metadata)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the module library."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Registry storage
        self._encoders: Dict[str, ModuleEntry] = {}
        self._decoders: Dict[str, ModuleEntry] = {}
        
        # Statistics tracking
        self._registration_count = 0
        self._selection_history: List[Dict[str, Any]] = []
        
        self.logger.info("ModuleLibrary initialized")
        
    def register_module(
        self, 
        module: Union[EncoderInterface, DecoderInterface], 
        metadata: ModuleMetadata,
        registration_id: Optional[str] = None
    ) -> str:
        """
        Register a module with the library.
        
        This method adds a module to the registry along with its metadata,
        enabling it to be discovered and selected by the intelligent matching
        algorithms.
        
        Args:
            module: The encoder or decoder instance to register
            metadata: Structured metadata describing the module's capabilities
            registration_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            str: The registration ID for this module
            
        Raises:
            TypeError: If module doesn't implement the correct interface
            ValueError: If metadata is invalid or registration_id conflicts
            
        Example:
            # Register a medical ECG encoder
            ecg_metadata = ModuleMetadata(
                module_type='encoder',
                data_modality='timeseries',
                domain='medical',
                capabilities=['r_peak_detection']
            )
            reg_id = library.register_module(ecg_encoder, ecg_metadata)
        """
        # Validate module type
        if metadata.module_type == 'encoder':
            if not isinstance(module, EncoderInterface):
                raise TypeError("Module must implement EncoderInterface for encoder type")
            registry = self._encoders
        elif metadata.module_type == 'decoder':
            if not isinstance(module, DecoderInterface):
                raise TypeError("Module must implement DecoderInterface for decoder type")
            registry = self._decoders
        else:
            raise ValueError(f"Invalid module_type: {metadata.module_type}")
            
        # Generate registration ID if not provided
        if registration_id is None:
            registration_id = f"{metadata.module_type}_{self._registration_count:04d}"
            
        # Check for ID conflicts
        if registration_id in registry:
            raise ValueError(f"Registration ID '{registration_id}' already exists")
            
        # Create registry entry
        entry = ModuleEntry(
            module=module,
            metadata=metadata,
            registration_id=registration_id
        )
        
        # Store in appropriate registry
        registry[registration_id] = entry
        self._registration_count += 1
        
        self.logger.info(
            f"Registered {metadata.module_type} '{registration_id}': "
            f"{metadata.domain}/{metadata.data_modality}"
        )
        
        return registration_id

    def find_best_match(
        self,
        data_metadata: Dict[str, Any],
        performance_preference: str = 'balanced'
    ) -> Tuple[Optional[EncoderInterface], Optional[DecoderInterface]]:
        """
        Find the best matching encoder and decoder for given data characteristics.

        This method implements the core meta-cognitive reasoning capability by
        analyzing data characteristics and selecting the most appropriate modules
        from the registry. The selection considers multiple compatibility factors
        and performance preferences.

        Args:
            data_metadata: Dictionary describing the data characteristics
            performance_preference: 'accuracy', 'speed', 'balanced', 'memory'

        Returns:
            Tuple of (best_encoder, best_decoder) or (None, None) if no matches

        Example:
            data_meta = {
                'data_modality': 'timeseries',
                'signal_type': 'periodic',
                'domain': 'medical',
                'input_format': 'dataframe'
            }
            encoder, decoder = library.find_best_match(data_meta)
        """
        # Find best encoder
        best_encoder = self._find_best_encoder(data_metadata, performance_preference)

        # Find best decoder (considering encoder output if available)
        decoder_requirements = data_metadata.copy()
        if best_encoder:
            # Update requirements based on encoder output
            encoder_config = best_encoder.get_config()
            decoder_requirements['input_format'] = 'tensor'  # Standard encoder output

        best_decoder = self._find_best_decoder(decoder_requirements, performance_preference)

        # Log selection
        self._log_selection(data_metadata, best_encoder, best_decoder)

        return best_encoder, best_decoder

    def _find_best_encoder(
        self,
        data_metadata: Dict[str, Any],
        performance_preference: str
    ) -> Optional[EncoderInterface]:
        """Find the best matching encoder for the data."""
        if not self._encoders:
            self.logger.warning("No encoders registered in library")
            return None

        best_score = -1.0
        best_encoder = None

        for entry in self._encoders.values():
            score = self._calculate_compatibility_score(
                entry.metadata, data_metadata, performance_preference
            )

            if score > best_score:
                best_score = score
                best_encoder = entry.module

        if best_score < 0.3:  # Minimum compatibility threshold
            self.logger.warning(f"No suitable encoder found (best score: {best_score:.2f})")
            return None

        return best_encoder

    def _find_best_decoder(
        self,
        data_metadata: Dict[str, Any],
        performance_preference: str
    ) -> Optional[DecoderInterface]:
        """Find the best matching decoder for the data."""
        if not self._decoders:
            self.logger.warning("No decoders registered in library")
            return None

        best_score = -1.0
        best_decoder = None

        for entry in self._decoders.values():
            score = self._calculate_compatibility_score(
                entry.metadata, data_metadata, performance_preference
            )

            if score > best_score:
                best_score = score
                best_decoder = entry.module

        if best_score < 0.3:  # Minimum compatibility threshold
            self.logger.warning(f"No suitable decoder found (best score: {best_score:.2f})")
            return None

        return best_decoder

    def _calculate_compatibility_score(
        self,
        module_metadata: ModuleMetadata,
        data_metadata: Dict[str, Any],
        performance_preference: str
    ) -> float:
        """
        Calculate compatibility score between module and data characteristics.

        This method implements the core matching algorithm that considers multiple
        dimensions of compatibility to determine how well a module fits the data.

        Args:
            module_metadata: Metadata of the module being evaluated
            data_metadata: Characteristics of the data to be processed
            performance_preference: User's performance preference

        Returns:
            float: Compatibility score (0.0-1.0, higher is better)
        """
        score = 0.0
        weight_sum = 0.0

        # Data modality match (highest weight)
        if 'data_modality' in data_metadata:
            weight = 0.4
            if module_metadata.data_modality == data_metadata['data_modality']:
                score += weight * 1.0
            elif module_metadata.data_modality == 'general':
                score += weight * 0.5  # General modules are fallbacks
            weight_sum += weight

        # Signal type match
        if 'signal_type' in data_metadata:
            weight = 0.3
            if module_metadata.signal_type == data_metadata['signal_type']:
                score += weight * 1.0
            elif module_metadata.signal_type == 'general':
                score += weight * 0.5
            weight_sum += weight

        # Domain match
        if 'domain' in data_metadata:
            weight = 0.2
            if module_metadata.domain == data_metadata['domain']:
                score += weight * 1.0
            elif module_metadata.domain == 'general':
                score += weight * 0.6
            weight_sum += weight

        # Input format compatibility
        if 'input_format' in data_metadata:
            weight = 0.1
            if module_metadata.input_format == data_metadata['input_format']:
                score += weight * 1.0
            weight_sum += weight

        # Performance preference adjustment
        performance_bonus = self._calculate_performance_bonus(
            module_metadata, performance_preference
        )
        score += performance_bonus * 0.1

        # Apply module priority and compatibility score
        score *= module_metadata.compatibility_score
        score *= (1.0 + module_metadata.priority * 0.1)

        # Normalize by total weight
        if weight_sum > 0:
            score = score / weight_sum

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_performance_bonus(
        self,
        module_metadata: ModuleMetadata,
        performance_preference: str
    ) -> float:
        """Calculate bonus score based on performance preference."""
        profile = module_metadata.performance_profile

        if performance_preference == 'accuracy' and profile.get('accuracy') == 'high':
            return 0.2
        elif performance_preference == 'speed' and profile.get('latency') == 'low':
            return 0.2
        elif performance_preference == 'memory' and profile.get('memory') == 'low':
            return 0.2
        elif performance_preference == 'balanced':
            return 0.1  # Small bonus for any reasonable performance

        return 0.0

    def _log_selection(
        self,
        data_metadata: Dict[str, Any],
        selected_encoder: Optional[EncoderInterface],
        selected_decoder: Optional[DecoderInterface]
    ) -> None:
        """Log module selection for analysis and debugging."""
        selection_record = {
            'data_metadata': data_metadata.copy(),
            'selected_encoder': type(selected_encoder).__name__ if selected_encoder else None,
            'selected_decoder': type(selected_decoder).__name__ if selected_decoder else None,
            'timestamp': self._get_timestamp()
        }

        self._selection_history.append(selection_record)

        self.logger.info(
            f"Module selection: Encoder={selection_record['selected_encoder']}, "
            f"Decoder={selection_record['selected_decoder']} for "
            f"{data_metadata.get('domain', 'unknown')}/{data_metadata.get('data_modality', 'unknown')}"
        )

    def get_registered_modules(self) -> Dict[str, List[str]]:
        """Get summary of all registered modules."""
        return {
            'encoders': list(self._encoders.keys()),
            'decoders': list(self._decoders.keys()),
            'total_count': len(self._encoders) + len(self._decoders)
        }

    def get_module_info(self, registration_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific module."""
        # Check encoders
        if registration_id in self._encoders:
            entry = self._encoders[registration_id]
        # Check decoders
        elif registration_id in self._decoders:
            entry = self._decoders[registration_id]
        else:
            return None

        return {
            'registration_id': entry.registration_id,
            'module_type': entry.metadata.module_type,
            'module_class': type(entry.module).__name__,
            'metadata': entry.metadata,
            'module_config': entry.module.get_config() if hasattr(entry.module, 'get_config') else {}
        }

    def unregister_module(self, registration_id: str) -> bool:
        """Remove a module from the registry."""
        if registration_id in self._encoders:
            del self._encoders[registration_id]
            self.logger.info(f"Unregistered encoder '{registration_id}'")
            return True
        elif registration_id in self._decoders:
            del self._decoders[registration_id]
            self.logger.info(f"Unregistered decoder '{registration_id}'")
            return True
        else:
            self.logger.warning(f"Registration ID '{registration_id}' not found")
            return False

    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get history of module selections for analysis."""
        return self._selection_history.copy()

    def clear_selection_history(self) -> None:
        """Clear the selection history."""
        self._selection_history.clear()
        self.logger.info("Selection history cleared")

    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        encoder_domains = {}
        decoder_domains = {}

        for entry in self._encoders.values():
            domain = entry.metadata.domain
            encoder_domains[domain] = encoder_domains.get(domain, 0) + 1

        for entry in self._decoders.values():
            domain = entry.metadata.domain
            decoder_domains[domain] = decoder_domains.get(domain, 0) + 1

        return {
            'total_modules': len(self._encoders) + len(self._decoders),
            'encoder_count': len(self._encoders),
            'decoder_count': len(self._decoders),
            'encoder_domains': encoder_domains,
            'decoder_domains': decoder_domains,
            'selection_count': len(self._selection_history),
            'registration_count': self._registration_count
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        import datetime
        return datetime.datetime.now().isoformat()

    def __repr__(self) -> str:
        """String representation of the module library."""
        stats = self.get_library_stats()
        return (
            f"ModuleLibrary(\n"
            f"  encoders: {stats['encoder_count']}\n"
            f"  decoders: {stats['decoder_count']}\n"
            f"  domains: {set(list(stats['encoder_domains'].keys()) + list(stats['decoder_domains'].keys()))}\n"
            f"  selections: {stats['selection_count']}\n"
            f")"
        )

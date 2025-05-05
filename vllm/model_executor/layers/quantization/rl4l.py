import itertools
from typing import Any, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase, 
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.fake_compress.quant import fake_quantize_weight

logger = init_logger(__name__)

class FakeCompConfig(QuantizationConfig):
    """Config class for fake compression."""

    def __init__(
            self, 
            init_bit_width: int = 16,
            modules_to_not_convert: Optional[List[str]] = None
        ) -> None:
        super().__init__()
        self.bit_width = init_bit_width
        self.modules_to_not_convert = modules_to_not_convert or []
    
    def __repr__(self) -> str:
        return (f"FakeCompConfig(bit_width={self.bit_width})")

    @classmethod
    def get_name(cls) -> str:
        return "fake_comp"
        
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.float, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75 #NOTE(brian1009): I just randomely put a lower number.
    
    @staticmethod
    def get_config_filenames() -> List[str]: # NOTE(brian1009): We don't need this, just a placeholder
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FakeCompConfig":
        return cls(config.get("bit_width", 16))

    def get_quant_method(self, layer: torch.nn.Module,
                        prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_fake_comp(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return FakeCompressedLinearMethod(self.bit_width)
        return None

def is_layer_skipped_fake_comp(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)

class FakeCompressedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def __init__(self, init_bit_width: int = 16):
        super().__init__()            # base classes don’t define __init__,
                                      # but the call does no harm and is good practice
        self.bit_width = init_bit_width    # <- remember the target bit-width

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
    
    def set_bit_width(self, bit_width: int):
        self.bit_width = bit_width
    
    def get_bit_width(self):
        return self.bit_width

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Patched version of apply that performs fake quantization before linear operation.
        
        Args:
            self: The UnquantizedLinearMethod instance
            layer: The linear layer
            x: Input tensor
            bias: Optional bias tensor
            
        Returns:
            Output of the linear operation with quantized weights
        """
        #logger.warning("Using patched apply")
        # Use the configured bit width from the quant_method
        bit_width = getattr(self, "bit_width", 16)  # Default to 8 if not set
        sym = getattr(self, "symmetric", True)  # Default to symmetric quantization
        group_size = getattr(self, "group_size", 128)  # Default group size
        clip_ratio = getattr(self, "clip_ratio", 1.0)  # Default clip ratio
        #NOTE(brian1009): We can extend this function to support more fake-compression methods
        w_compressed = fake_quantize_weight(
            layer.weight, 
            n_bits=bit_width,
            sym=sym,
            group_size=group_size,
            clip_ratio=clip_ratio
        )
        return F.linear(x, w_compressed, bias)
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.parameter import ModelWeightParameter


class TorchAOMethod(QuantizeMethodBase):
    """Implements quantization method for TorchAO."""

    def __init__(self, config):
        self.config = config
        self.weight_bits = (config.bit_width
                            )  # Changed from weight_bits to bit_width
        self.group_size = (
            128  # Added default group size, should be configurable
        )
        self.scheme = config.scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create quantized weights for the layer."""
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        # weight_dtype = (
        #     torch.float8_e4m3fn
        #     if self.quant_config.is_checkpoint_fp8_serialized
        #     else params_dtype
        # )
        # torch.distributed.breakpoint()
        from torchao.quantization.quant_api import quantize_,Int8WeightOnlyConfig


        weight_dtype = layer.params_dtype

        tst = torch.nn.Linear(input_size_per_partition, output_size_per_partition, bias=False)
        quantize_(tst, Int8WeightOnlyConfig())
        weight = ModelWeightParameter(
            data=tst.weight,
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # if self.quant_config.is_checkpoint_fp8_serialized:
        #     # WEIGHT SCALE
        #     weight_scale = PerTensorScaleParameter(data=torch.empty(
        #         len(output_partition_sizes), dtype=torch.float32),
        #                                            weight_loader=weight_loader)
        #     weight_scale[:] = torch.finfo(torch.float32).min
        #     layer.register_parameter("weight_scale", weight_scale)
        #     # INPUT SCALE
        #     scale = PerTensorScaleParameter(data=torch.empty(
        #         len(output_partition_sizes), dtype=torch.float32),
        #                                     weight_loader=weight_loader)

        #     scale[:] = torch.finfo(torch.float32).min
        #     layer.register_parameter("input_scale", scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the quantized weights to input."""
        weight = layer.weight
        breakpoint()
        if hasattr(layer, "scale"):
            # Reshape scale for broadcasting
            scale = layer.scale.view(-1, 1)
            weight = weight * scale

            if hasattr(layer, "zero_point"):
                zero_point = layer.zero_point.view(-1, 1)
                weight = weight + zero_point

        output = F.linear(x, weight, bias)
        return output


class TorchAOConfig(QuantizationConfig):

    def __init__(
            self,
            bit_width: int = 8,
            use_symmetric: bool = True,
            scheme: str = "weight_only",
            calibration: str = "minmax",
            per_channel: bool = True,
            observer_type: str = "minmax",
            group_size: int = 128,  # Added group_size parameter
    ):
        super().__init__()
        self.bit_width = bit_width
        self.use_symmetric = use_symmetric
        self.scheme = scheme
        self.calibration = calibration
        self.per_channel = per_channel
        self.observer_type = observer_type
        self.group_size = group_size  # Added group_size attribute

    def apply_quantization(self, model):
        from torchao.quantization.quant_api import (Float8WeightOnlyConfig,
                                                    Int4WeightOnlyConfig,
                                                    Int8WeightOnlyConfig,
                                                    quantize_)

        if self.bit_width == 4:
            config = Int4WeightOnlyConfig()
        elif self.bit_width == 8:
            config = Int8WeightOnlyConfig()
        elif self.bit_width == 16:
            config = Float8WeightOnlyConfig()
        else:
            raise ValueError(f"Unsupported bit_width: {self.bit_width}")

        quantize_(model, config)

    def validate_config(self):
        if self.bit_width not in [4, 8, 16]:
            raise ValueError("Invalid bit width for TorchAO quantization")
        if not isinstance(self.use_symmetric, bool):
            raise ValueError("use_symmetric must be a boolean value")
        if not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")

    def get_name(self) -> str:
        return "torchao"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["torchao.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TorchAOConfig":
        bit_width = config.get("bit_width", 8)
        use_symmetric = config.get("use_symmetric", True)
        group_size = config.get("group_size", 128)  # Added group_size
        return cls(
            bit_width=bit_width,
            use_symmetric=use_symmetric,
            group_size=group_size,
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return TorchAOMethod(self)
        elif isinstance(layer, Attention):
            return None
        return None

"""WrinkleFree Inference Engine - SGLang-BitNet serving with native SIMD kernels."""

__version__ = "0.1.0"

# Primary modules (SGLang-BitNet stack)
from wrinklefree_inference.client.bitnet_client import BitNetClient
from wrinklefree_inference.sglang_backend import quantize_to_bitnet, pack_weights_bitnet

__all__ = ["BitNetClient", "quantize_to_bitnet", "pack_weights_bitnet", "__version__"]

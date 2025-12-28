"""Test KV Cache C++ implementation vs Python reference.

This test file validates that C++ KV cache operations match Python reference
implementations. Tests are added incrementally as C++ features are implemented.

Iteration 1: Basic allocation tests (Python reference only)
Iteration 2-3: Gather/scatter tests (after C++ implementation)
Iteration 4+: Full integration tests (after Python bindings)
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Python Reference Implementations
# =============================================================================

class PythonKVCacheReference:
    """Pure Python KV cache reference implementation for correctness testing."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype

        # Storage: [max_pages, num_layers, 2, num_heads, head_dim]
        # 2 = K and V
        self.cache_storage = torch.zeros(
            max_pages, num_layers, 2, num_heads, head_dim, dtype=dtype
        )

        # Free page list
        self._free_pages = list(range(max_pages - 1, -1, -1))

    def allocate_page(self) -> int:
        """Allocate a page. Returns -1 if no pages available."""
        if not self._free_pages:
            return -1
        return self._free_pages.pop()

    def allocate_pages(self, num_pages: int) -> torch.Tensor:
        """Allocate multiple pages. Returns empty tensor if not enough."""
        if len(self._free_pages) < num_pages:
            return torch.empty(0, dtype=torch.int32)
        pages = [self._free_pages.pop() for _ in range(num_pages)]
        return torch.tensor(pages, dtype=torch.int32)

    def free_page(self, page_id: int):
        """Return a page to the free list."""
        if 0 <= page_id < self.max_pages:
            self._free_pages.append(page_id)

    def free_pages_batch(self, page_ids: torch.Tensor):
        """Return multiple pages to the free list."""
        for page_id in page_ids.tolist():
            self.free_page(page_id)

    def num_free_pages(self) -> int:
        """Get number of free pages."""
        return len(self._free_pages)

    def gather_kv(
        self,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K and V from cache.

        Args:
            page_indices: [num_tokens] page indices
            slot_indices: [num_tokens] slot within page (unused in Iter 1)
            layer_id: Layer index

        Returns:
            k_out: [num_tokens, num_heads, head_dim]
            v_out: [num_tokens, num_heads, head_dim]
        """
        num_tokens = page_indices.shape[0]
        k_out = torch.zeros(num_tokens, self.num_heads, self.head_dim, dtype=self.dtype)
        v_out = torch.zeros(num_tokens, self.num_heads, self.head_dim, dtype=self.dtype)

        for t in range(num_tokens):
            page_id = page_indices[t].item()
            # K = cache[page, layer, 0, :, :]
            # V = cache[page, layer, 1, :, :]
            k_out[t] = self.cache_storage[page_id, layer_id, 0]
            v_out[t] = self.cache_storage[page_id, layer_id, 1]

        return k_out, v_out

    def scatter_kv(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ):
        """Scatter K and V to cache.

        Args:
            k_in: [num_tokens, num_heads, head_dim]
            v_in: [num_tokens, num_heads, head_dim]
            page_indices: [num_tokens] page indices
            slot_indices: [num_tokens] slot within page (unused in Iter 1)
            layer_id: Layer index
        """
        num_tokens = page_indices.shape[0]

        for t in range(num_tokens):
            page_id = page_indices[t].item()
            self.cache_storage[page_id, layer_id, 0] = k_in[t]
            self.cache_storage[page_id, layer_id, 1] = v_in[t]


# =============================================================================
# Test Configuration
# =============================================================================

# BitNet-b1.58-2B-4T dimensions
BITNET_CONFIG = {
    "num_layers": 32,
    "num_heads": 20,
    "head_dim": 128,
    "page_size": 256,
    "max_pages": 100,
}


# =============================================================================
# Iteration 1 Tests: Basic Allocation
# =============================================================================

class TestPythonReference:
    """Test Python reference implementation (used as ground truth)."""

    def test_init(self):
        """Test cache initialization."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"]
        assert cache.cache_storage.shape == (
            BITNET_CONFIG["max_pages"],
            BITNET_CONFIG["num_layers"],
            2,  # K, V
            BITNET_CONFIG["num_heads"],
            BITNET_CONFIG["head_dim"],
        )

    def test_allocate_single(self):
        """Test single page allocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        page1 = cache.allocate_page()
        assert 0 <= page1 < BITNET_CONFIG["max_pages"]
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 1

        page2 = cache.allocate_page()
        assert page2 != page1
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 2

    def test_allocate_batch(self):
        """Test batch page allocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        pages = cache.allocate_pages(10)
        assert pages.shape == (10,)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 10

        # All pages should be unique
        assert len(set(pages.tolist())) == 10

    def test_allocate_too_many(self):
        """Test allocation failure when not enough pages."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        # Allocate all pages
        all_pages = cache.allocate_pages(BITNET_CONFIG["max_pages"])
        assert all_pages.shape == (BITNET_CONFIG["max_pages"],)

        # Try to allocate more
        more_pages = cache.allocate_pages(1)
        assert more_pages.shape == (0,)

        single = cache.allocate_page()
        assert single == -1

    def test_free_pages(self):
        """Test page deallocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        pages = cache.allocate_pages(5)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 5

        cache.free_pages_batch(pages)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"]


class TestGatherScatterReference:
    """Test gather/scatter in Python reference."""

    def test_scatter_then_gather(self):
        """Test roundtrip: scatter then gather."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)

        # Allocate some pages
        num_tokens = 8
        pages = cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)  # Not used in Iter 1
        layer_id = 5

        # Create random K/V
        k_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])

        # Scatter to cache
        cache.scatter_kv(k_in, v_in, pages, slots, layer_id)

        # Gather back
        k_out, v_out = cache.gather_kv(pages, slots, layer_id)

        # Should match
        assert torch.allclose(k_out, k_in, atol=1e-5)
        assert torch.allclose(v_out, v_in, atol=1e-5)

    def test_gather_different_layers(self):
        """Test that different layers are independent."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)

        num_tokens = 4
        pages = cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)

        # Scatter to layer 0
        k0 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v0 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        cache.scatter_kv(k0, v0, pages, slots, layer_id=0)

        # Scatter different values to layer 1
        k1 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v1 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        cache.scatter_kv(k1, v1, pages, slots, layer_id=1)

        # Gather from both layers
        k0_out, v0_out = cache.gather_kv(pages, slots, layer_id=0)
        k1_out, v1_out = cache.gather_kv(pages, slots, layer_id=1)

        # Layer 0 should match original layer 0 data
        assert torch.allclose(k0_out, k0, atol=1e-5)
        assert torch.allclose(v0_out, v0, atol=1e-5)

        # Layer 1 should match original layer 1 data
        assert torch.allclose(k1_out, k1, atol=1e-5)
        assert torch.allclose(v1_out, v1, atol=1e-5)


# =============================================================================
# Iteration 4+ Tests: C++ vs Python Comparison
# =============================================================================

class TestCppVsPython:
    """Compare C++ implementation against Python reference.

    These tests are skipped until C++ bindings are implemented (Iteration 4).
    """

    @pytest.fixture
    def cpp_available(self):
        """Check if C++ KV cache bindings are available."""
        try:
            from sgl_kernel.kvcache import (
                kv_cache_create,
                kv_cache_gather,
                kv_cache_scatter,
            )
            return True
        except ImportError:
            return False

    @pytest.mark.skip(reason="C++ bindings not implemented until Iteration 4")
    def test_gather_cpp_vs_python(self, cpp_available):
        """Test C++ gather matches Python reference."""
        if not cpp_available:
            pytest.skip("C++ KV cache bindings not available")

        # Will be implemented in Iteration 4
        pass

    @pytest.mark.skip(reason="C++ bindings not implemented until Iteration 4")
    def test_scatter_cpp_vs_python(self, cpp_available):
        """Test C++ scatter matches Python reference."""
        if not cpp_available:
            pytest.skip("C++ KV cache bindings not available")

        # Will be implemented in Iteration 4
        pass

    @pytest.mark.skip(reason="C++ bindings not implemented until Iteration 4")
    def test_allocation_cpp_vs_python(self, cpp_available):
        """Test C++ allocation matches Python behavior."""
        if not cpp_available:
            pytest.skip("C++ KV cache bindings not available")

        # Will be implemented in Iteration 4
        pass

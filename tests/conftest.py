"""Configuration for pytest, including fixtures and R availability detection."""
import pytest


def pytest_configure(config):
    """Detect if R dependencies are available."""
    try:
        import rpy2
        pytest.R_AVAILABLE = True
    except ImportError:
        pytest.R_AVAILABLE = False
        print("\nWarning: R dependencies (rpy2) not available. R-based tests will be skipped.")

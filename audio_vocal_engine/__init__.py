"""Standalone, process-isolated vocal processing engine.

Importing the package and inspecting the catalog uses only the standard library.
Heavy inference dependencies are imported after the shared engine lock is held.
"""

__version__ = "1.0.0"

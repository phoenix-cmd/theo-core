"""Single source of truth for THEO version information.

All runtime version strings derive from this module. The build backend
(hatchling) reads ``__version__`` from this file via ``[tool.hatch.version]``.
"""

__version__ = "0.4.1"

VERSION = __version__

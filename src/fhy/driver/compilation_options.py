"""Option definitions for compilation of FhY source code."""

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class CompilationOptions:
    """Supported FhY compilation options."""

    verbose: bool = field(default=False)

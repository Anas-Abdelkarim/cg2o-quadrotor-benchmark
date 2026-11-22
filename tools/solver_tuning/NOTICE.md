# Provenance

This package was developed from an earlier internal `tuning` library. It replaces that library's architecture while
retaining the idea of typed parameter domains and explicit experiment adapters. Input configurations are never
overwritten.

Optuna, PyYAML and optional plotting/build/test dependencies remain separate third-party packages with their
respective licenses. This package does not bundle their source or assign a new license to the code it tunes.

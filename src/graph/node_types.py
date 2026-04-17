"""Joern node-type vocabulary (no torch / PyG). Shared by dataset prep and GNN code."""

DEFAULT_X_TYPES_RAW: list[str] = [
    "BLOCK",
    "CALL",
    "CONTROL_STRUCTURE",
    "FIELD_IDENTIFIER",
    "IDENTIFIER",
    "JUMP_TARGET",
    "LITERAL",
    "LOCAL",
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_PARAMETER_OUT",
    "METHOD_REF",
    "METHOD_RETURN",
    "MODIFIER",
    "RETURN",
    "TYPE_DECL",
    "UNKNOWN",
]

DEFAULT_X_TYPES: dict[str, int] = {t: i for i, t in enumerate(DEFAULT_X_TYPES_RAW)}

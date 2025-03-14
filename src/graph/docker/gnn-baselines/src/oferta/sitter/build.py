from pathlib import Path

from tree_sitter import Language


APP_ROOT = Path("/gnn-baselines")
TREE_SITTER_LIB = APP_ROOT / "build" / "tree-sitter.so"
# TREE_SITTER_LIB.parent.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    Language.build_library(
        # Store the library in the `build` directory
        str(TREE_SITTER_LIB),
        # Include one or more languages
        [
            "/opt/tree-sitter-c"
        ],
    )

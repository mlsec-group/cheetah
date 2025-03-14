from pathlib import Path

from tree_sitter import Language, Parser, Tree
import tree_sitter_c as ts_c


Q_FUNCTION_DEF = """(
((function_definition) @func)
)"""

Q_FUNCTION_NAME = """
(function_declarator
    declarator: ((identifier) @name)
)
"""


class TreeSitter(object):
    def __init__(self, lib_path: Path):
        self.c_lang = Language(ts_c.language())
        self.parser = Parser(self.c_lang)

        self.q_function_def = self.c_lang.query(Q_FUNCTION_DEF)
        self.q_function_name = self.c_lang.query(Q_FUNCTION_NAME)

    def parse(self, code: str) -> Tree:
        return self.parser.parse(bytes(code, "utf8"))

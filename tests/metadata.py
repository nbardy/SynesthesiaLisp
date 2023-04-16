

import lark
import re
from lark import Lark
from lark import Lark, Transformer, v_args

metadata_grammar = """
?start: metadata_dict

metadata_dict: "{" key_value_pair_list "}"
key_value_pair_list: key_value_pair (","? key_value_pair)*
key_value_pair: key value
key: SYMBOL

value: string | number | list | metadata_dict

string: DOUBLE_QUOTED_STRING
DOUBLE_QUOTED_STRING: /"(?:[^"\\\\]|\\\\.|\\n)*"/s
number: SIGNED_INT | FLOAT

list: "[" value_list "]"
value_list: (value ","?)*

SYMBOL: /:[A-Za-z][-A-Za-z0-9_]*/i
ESCAPED_STRING: /"(\\"|[^"])*"/
SIGNED_INT: /-?\d+/i
FLOAT: /-?\d+\.\d+/i

%import common.WS
%ignore WS
"""


class MetadataTransformer(Transformer):
    @v_args(inline=True)
    def metadata_dict(self, items):
        return self._process_dict({key: value for key, value in items})

    def key_value_pair_list(self, items):
        return items

    def key_value_pair(self, items):
        return items[0], items[1]

    def key(self, items):
        return str(items[0])

    def string(self, items):
        return items[0][1:-1]

    def number(self, items):
        num_str = items[0]
        return int(num_str) if num_str.isdigit() else float(num_str)

    def list(self, items):
        return self._process_list(items)

    def value_list(self, items):
        return items

    def _process_dict(self, d):
        # return {key: self._process_value(value) for key, value in d.items()}
        # Remove ":" leader
        return {key[1:]: self._process_value(value) for key, value in d.items()}

    def _process_list(self, items):
        return self._process_value(items[0])

    def _process_value(self, value):
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        elif isinstance(value, list):
            return [self._process_value(v) for v in value]
        elif isinstance(value, dict):
            return self._process_dict(value)

        elif isinstance(value, lark.tree.Tree):
            return self._process_value(value.children[0])
        else:
            raise ValueError(f"Unknown value type: {type(value)}")


metadata_parser = Lark(metadata_grammar, parser='lalr',
                       transformer=MetadataTransformer())


def parse_metadata(metadata_str):
    return metadata_parser.parse(metadata_str)


metadata_example = """
(let 
    ^{:model "blip2"
    :model-args ["test_example.jpg" "test_example_2.jpg"]
    [tree "Big"] tree)
"""

metadata_str = metadata_example.strip().lstrip('^')
metadata_dict = parse_metadata(metadata_str)
print(metadata_dict)

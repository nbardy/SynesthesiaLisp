from lark import Lark, Tree, Token
input_str = '''
(let
 ^{:model "blip2"
  :model-args ["test_example.jpg"]}
  [people-count "How many people are in the image, answer zero if none?"]
  tree)
'''

grammar = 'src/sample.lark'

with open(grammar, 'r') as f:
    parser = Lark(f, start='start')

tree = parser.parse(input_str)
print(tree.pretty())

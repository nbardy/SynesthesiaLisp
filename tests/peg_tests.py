from parsimonious.grammar import Grammar
from parsimonious.exceptions import ParseError

# Load the grammar from a file
with open("src/super_prompt.lark") as f:
    grammar = Grammar(f.read())

# Define your input
input_string = '''
(let
 ^{:model "blip2"
   :model-args ["test_example.jpg"]}
 [people-count "How many people are in the image, answer zero if none?"
  subject (cond
            (= people-count 0) "What is in this image?"
            (= people-count 1) "Q: What does the person in this image look like? 
                                 Notes: Add comments on the most interesting features, what is visualy stunning about this person?
                                 A:"
            (>= people-count 2) "Q: What does the person in this image look like? 
                                 Notes: Add comments on the most interesting features, what is visualy stunning about this person?
                                 A:")
  total-caption (str "The Subject is ")
  critique-caption "Critique the caption: {subject}"]
  "Given the following:
    caption
    {total-caption}

    critique caption:
    {critique-caption}

    Generate a final caption densely that describes the most interesting parts of the image in the simplest way.")
'''

try:
    parse_tree = grammar.parse(input_string)
    print("Parsed successfully!")
except ParseError as e:
    print("Parsing failed:", e)

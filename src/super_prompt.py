from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import argparse
from alive_progress.animations.spinners import bouncing_spinner_factory, sequential_spinner_factory
from alive_progress import alive_bar
from alive_progress import config_handler
import ast
from alive_progress import alive_bar, config_handler
from lark import Tree, Token
import lark
import re
from lark import Lark
from lark import Lark, Transformer, v_args

import openai

# Add your OpenAI API key here

openai.api_key = "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"
openai.organization = "org-9FuYsKOwtUzcDFxwPQBCgHe1"

device = "cuda" if torch.cuda.is_available() else "cpu"


# Useful for testing non gpu
default_model_context_key = "openai_chat_turbo"


def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".webp"])


def process_args(args):
    # Take the string of vector and make a list
    # ["file.jpg", "tree"]
    args = args.split(",")
    args = [arg.strip() for arg in args]

    # Now open all image files and convert to PIL
    # ["file.jpg", "tree"] -> [PIL.Image, "tree"]
    for i, arg in enumerate(args):
        if is_image(arg):
            args[i] = Image.open(arg)


def send_to_llm(content, env):
    current_llm_metadata = env.get(
        MODEL_CONTEXT_KEY, default_model_context_key)
    print(" CURRENT")
    print(current_llm_metadata)
    print(" CURRETN")
    model = current_llm_metadata["model"]
    model_args = current_llm_metadata["model-args"]

    if model == "openai_chat_turbo":
        return eval_openai_chat_turbo(content, env)
    elif model == "blip2":
        return eval_blip_local(content, process_args(model_args))[0]
    else:
        raise ValueError(f'Unknown Model: {model}')


# Currently used for testing on GPU machines
def eval_openai_chat_turbo(content):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
    )

    assistant_reply = response.choices[0].message['content']
    return assistant_reply


def load_blip2_0(model_id="Salesforce/blip2-flan-t5-xxl"):
    """
    Function to load and cache the BLIP 2.0 processor and model.

    Args:
        model_id (str, optional): The Hugging Face model ID for the BLIP 2.0 model. Defaults to "Salesforce/blip2-flan-t5-xxl".

    Returns:
        tuple: A tuple containing the Blip2Processor and Blip2ForConditionalGeneration instances.
    """

    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16)
    model.to(device)

    return processor, model


def eval_blip_local(content, image):
    """
    Function to evaluate the BLIP 2.0 model on the given text and image inputs.

    Args:
        content (str): The text input for the BLIP 2.0 model.
        image (PIL.Image.Image): The image input for the BLIP 2.0 model.

    Returns:
        str: The generated response from the BLIP 2.0 model.
    """
    # Ensure the image is in RGB format
    image = image.convert('RGB')

    processor, model = load_blip2_0()

    # Process the inputs using the Blip2Processor
    inputs = processor(images=image, text=content,
                       return_tensors="pt").to(device, torch.float16)

    # Generate the output using the Blip2ForConditionalGeneration model
    generated_ids = model.generate(**inputs)

    # Decode the generated output and return it as a string
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


def build_str(s: str, env: dict) -> str:
    def get_value(token: str, include_prompt: bool = False) -> str:
        env_obj = env.get(token)

        if not env_obj:
            return f"{{{token}}}"

        if env_obj["type"] == "StringUnparsed":
            return build_str(env_obj["value"], env) if not include_prompt else f"{env_obj['value']}"

        elif env_obj["type"] == "StringValue":
            return env_obj["value"]

        elif env_obj["type"] == "StringGen":
            return env_obj["value"] if not include_prompt else f"{env_obj['prompt']}{env_obj['value']}"

        else:
            raise ValueError(f"Invalid type: {env_obj['type']}")

    result = ""
    idx = 0
    while idx < len(s):
        if s[idx] == "{":
            token_end = s.find("}", idx + 1)
            if token_end == -1:
                result += s[idx:]
                break

            token = s[idx + 1: token_end]
            include_prompt = False

            if ":" in token:
                token, modifier = token.split(":", 1)
                if modifier == "full":
                    include_prompt = True

            result += get_value(token, include_prompt)
            idx = token_end + 1

        else:
            result += s[idx]
            idx += 1

    return result


def eval_str(item, env):
    text = item.children[0].value
    text = build_str(text, env)

    print("🪄 Querying the LLM")
    print("🪄 Query: \n\n", text, "\n")

    # Custom emoji animation
    magic_emojis = bouncing_spinner_factory("🌈🔮✨💫🪄🌟", 12)
    magic_spinner = sequential_spinner_factory(*[magic_emojis]*12)

    # Configure the magic theme for the animated loader
    # config_handler.set_global(theme='smooth', spinner=magic_spinner)

    with alive_bar(title="🔮🪄 Sending query with magic... ", spinner=magic_spinner) as bar:
        llm_response = send_to_llm(text, env)
        bar()

    print("🌐 Response: \n\n", llm_response, "\n")

    return llm_response


def eval_ast(expr, env):
    if expr.data == 'start' or expr.data == 'expr':
        return eval_ast(expr.children[0], env)
    elif expr.data == 'name':
        return eval_name(expr, env)
    elif expr.data == 'val':
        return eval_val(expr, env)
    elif expr.data == 'string':
        return eval_str(expr, env)
    elif expr.data == 'sexp':
        wrapped_expr = expr.children[0]
        return eval_sexpr(wrapped_expr, env)
    else:
        raise ValueError(f'Unknown expression: {expr}')

# Split metadat
# if first child looks like this
# Binding: Tree(Token('RULE', 'metadata'), [Token('CARET_METADATA', '^{:model "blip2"\n   :model-args ["test_example.jpg"]}')])
# Get the first child of the metadata tree
# parse it with the metadata parser
# If first child is no there return none
# Return rest of children as well


def split_metadata(tree: Tree):
    if len(tree.children) == 0:
        raise ValueError("No children in tree")

    print("== tre")
    print(tree)
    print("== child")
    first_child = tree.children[0]
    print(first_child.data)
    if first_child.data == "metadata":
        metadata = first_child.children[0].value
        parsed_metadata = parse_metadata(metadata)

        return parsed_metadata, tree.children[1:]

    return None, tree.children


def parse_metadata(metadata_str):
    return metadata_parser.parse(metadata_str)


MODEL_CONTEXT_KEY = "__model_context_key__"


def eval_let_form(tree: Tree, context: dict):
    new_context = context.copy()
    print("Eval Let =========")
    print("tree: ")
    print(tree)

    metadata, children = split_metadata(tree)

    # Add metadata to context
    if metadata:
        new_context[MODEL_CONTEXT_KEY] = metadata

    for binding in children:
        # Add this line for debugging purposes
        print("===== binding Hello =====")
        print("===== binding =====")
        print(f"Binding: {binding}")
        print("Length", len(binding.children))
        # Binding: Tree('binding', [Tree(Token('RULE', 'name'), ['emotion']), Tree(Token('RULE', 'expr'), [
        #               Tree(Token('RULE', 'string'), ['Describe the emotional content of this scene'])])])
        if len(binding.children) == 2:
            name_tree = binding.children[0]
            val_tree = binding.children[1]

            # Check that the data is name
            # Then get the first child of the name tree
            if not name_tree.data == "name":
                raise ValueError("Invalid binding format")

            name = name_tree.children[0]

            value = eval_ast(val_tree, new_context)
            new_context[name] = value
        else:
            raise ValueError("Invalid binding format")
    return eval_ast(tree.children[1], new_context)


SNAME_RE = r"[a-zA-Z\-_0-9:]+"


with open("src/super_prompt.lark", "r") as file:
    grammar = file.read()


dsl_parser = Lark(grammar, start="start", parser="lalr")


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


# Define a transformer class to convert the AST into a more convenient format


class SuperPromptTransformer(Transformer):
    @ v_args(inline=True)
    def sexpr(self, items):
        return [items]

    def SNAME(self, token):
        try:
            return str(token)
        except KeyError as e:
            raise ValueError(f"Undefined token: {e}")

    def STRING(self, token):
        return ast.literal_eval(token)

    def binding(self, items):
        return Tree("binding", [items[0], items[1]])

    # unwrap start

    def start(self, items):
        return items[0]


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


# Parse the DSL code


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


def eval_case(ast, env):
    test_value = eval_ast(ast[0], env)
    cases = ast[1:]
    for i in range(0, len(cases), 2):
        case_value = eval_ast(cases[i], env)
        if test_value == case_value:
            return eval_ast(cases[i + 1], env)
    if len(cases) % 2 == 1:
        return eval_ast(cases[-1], env)
    return None


def eval_loose_match_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    return operand1 == operand2


def explain_error(explanation, error):
    print("👇==== Error ======")
    print("Explanation: ")
    print(explanation)
    print("☝===== Error ======")
    raise error


# It casts to boolean for
# Also trims whitespace and is case insensitive
# true: "yes", "y", "true", "t", "1", "1.0", "on", "enable", "enabled"
# false: "no", "n", "false", "f", "0", "0.0", "", "none", "[]", "{}", "", "null", "nil", "na", "nan", "N/A"

truthy_values = ["yes", "y", "true", "t",
                 "1", "1.0", "on", "enable", "enabled"]
falsey_values = ["no", "n", "false", "f", "0", "0.0", "none",
                 "[]", "{}", "", "null", "nil", "na", "nan", "N/A"]


# Defaults to false
def loose_boolean_conformity(value):
    value = value.strip().lower()

    if value in truthy_values:
        return True
    elif value in falsey_values:
        return False


def eval_question_form(expr, env):

    # Eval child then "see if it matches"
    val = eval_ast(expr.children[0], env)
    print("val")
    print(val)

    val = loose_boolean_conformity(val)

    return val


def eval_cond_form(expr, env):
    for clause in expr.children:
        condition = eval_ast(clause.children[0], env)
        if condition:
            return eval_ast(clause.children[1], env)
    return None


def eval_case_form(expr, env):
    key = eval_ast(expr.children[0], env)
    for clause in expr.children[1:]:
        if key == clause.children[0].value:
            return eval_ast(clause.children[1], env)
    return None


def eval_if_form(expr, env):
    condition = eval_ast(expr.children[0], env)
    if condition:
        return eval_ast(expr.children[1], env)
    else:
        return eval_ast(expr.children[2], env)


def eval_math_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    op = expr.data
    if op == 'add':
        return operand1 + operand2
    elif op == 'sub':
        return operand1 - operand2
    elif op == 'mul':
        return operand1 * operand2
    elif op == 'div':
        return operand1 / operand2
    else:
        raise ValueError(f'Unknown math operation: {op}')


def eval_comp_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    op = expr.data
    if op == 'eq':
        return operand1 == operand2
    elif op == 'ne':
        return operand1 != operand2
    elif op == 'lt':
        return operand1 < operand2
    elif op == 'le':
        return operand1 <= operand2
    elif op == 'gt':
        return operand1 > operand2
    elif op == 'ge':
        return operand1 >= operand2
    else:
        raise ValueError(f'Unknown comparison operation: {op}')


def eval_str_form(expr, env):
    return ''.join(map(lambda child: str(eval_ast(child, env)), expr.children))


def eval_name(expr, env):
    print("== eval name ==")
    print(env)
    print(expr.children[0])

    return env[expr.children[0]]


def eval_val(expr, env):
    return int(expr.children[0].value)


eval_functions = {
    'if_form': eval_if_form,
    'math_form': eval_math_form,
    'comp_form': eval_comp_form,
    'loose_match_form': eval_loose_match_form,
    'question_form': eval_question_form,
    'cond_form': eval_cond_form,
    'case_form': eval_case_form,
    'let_form': eval_let_form,
    'str_form': eval_str_form,
}


def eval_sexpr(expr, env):
    form_name = expr.data
    return eval_functions[form_name](expr, env)


default = "examples/film_simple.super_prompt"

# argparse for file options
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default=default, help='file to run')

args = parser.parse_args()
file = args.file


with open(file, "r") as f:
    dsl_code = f.read()

dsl_code = """
(let 
    ^{:model "blip2"
    :model-args ["test_example.jpg" "test_example_2.jpg"]}
    [tree "Big"] tree)
"""


# Implement the eval_* functions for each form as needed.
ast = parse_super_prompt(dsl_code)
print("==== EVAL =====")
result = eval_ast(ast, {})
print("==== EVAL DONE =====")
print("==== RESULT =====")
# print(result)
print("==== END RESULT =====")


# ====================
# Version

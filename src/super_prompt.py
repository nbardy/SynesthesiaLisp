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


# Useful for testing non gpu
current_llm = "openai_chat_turbo"


def send_to_llm(content):
    if current_llm == "openai_chat_turbo":
        return eval_openai_chat_turbo(content)
    elif current_llm == "blip":
        return eval_blip_local(content)
    else:
        raise ValueError(f'Unknown LLM: {current_llm}')


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


def eval_blip_local(content):
    # TODO
    return


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
        llm_response = send_to_llm(text)
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


def eval_let_form(tree: Tree, context: dict):
    new_context = context.copy()
    print("Eval Let =========")
    print("tree: ")
    print(tree)
    for binding in tree.children:
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
    @v_args(inline=True)
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


# Parse the DSL code


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


with open("examples/film.super_prompt", "r") as f:
    dsl_code = f.read()


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


def eval_sexpr(expr, env):
    form_name = expr.data
    return eval_functions[form_name](expr, env)


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

import numpy as np
import ast
import operator


# ============================================================================
# Expression Validation and Compilation Functions
# ============================================================================

# Allowed operations for expression validation
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions
ALLOWED_FUNCTIONS = {
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "pow": np.power,
}


def normalize_expression(expr):
    """
    Normalize expression by replacing common math functions with numpy equivalents.

    Args:
        expr: String expression to normalize

    Returns:
        Normalized expression string
    """
    expr = expr.replace("sqrt", "np.sqrt")
    expr = expr.replace("sin(", "np.sin(")
    expr = expr.replace("cos(", "np.cos(")
    expr = expr.replace("tan(", "np.tan(")
    # Handle cases where np.np might occur
    expr = expr.replace("np.np.", "np.")
    return expr


def validate_expression(expr, allowed_vars):
    """
    Validate that expression only contains safe operations.

    Args:
        expr: String expression to validate
        allowed_vars: Set of allowed variable names

    Raises:
        ValueError: If expression contains dangerous operations
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr}") from e

    # Walk through AST and check all nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # Check variable names
            if node.id not in allowed_vars and node.id not in ALLOWED_FUNCTIONS:
                raise ValueError(
                    f"Disallowed variable '{node.id}' in expression: {expr}"
                )

        elif isinstance(node, ast.Call):
            # Check function calls
            if isinstance(node.func, ast.Attribute):
                # Handle np.sqrt, np.sin, etc.
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id != "np":
                        raise ValueError(
                            f"Disallowed module '{node.func.value.id}' in expression: {expr}"
                        )
                    # np.function_name - allow numpy functions
                else:
                    raise ValueError(f"Complex attribute access not allowed: {expr}")

            elif isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_FUNCTIONS:
                    raise ValueError(
                        f"Disallowed function '{node.func.id}' in expression: {expr}"
                    )
            else:
                raise ValueError(f"Complex function call not allowed: {expr}")

        elif isinstance(node, ast.BinOp):
            # Check binary operations
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError(
                    f"Disallowed operator '{node.op.__class__.__name__}' in expression: {expr}"
                )

        elif isinstance(node, ast.UnaryOp):
            # Check unary operations
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError(
                    f"Disallowed unary operator '{node.op.__class__.__name__}' in expression: {expr}"
                )

        elif isinstance(node, (ast.Attribute, ast.Import, ast.ImportFrom)):
            # Block imports and most attribute access
            if isinstance(node, ast.Attribute):
                # Only allow np.function
                if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
                    raise ValueError(
                        f"Disallowed attribute access in expression: {expr}"
                    )
            else:
                raise ValueError(f"Import statements not allowed in expression: {expr}")

        elif isinstance(node, (ast.Lambda, ast.FunctionDef, ast.ClassDef)):
            raise ValueError(
                f"Function/class definitions not allowed in expression: {expr}"
            )

        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):
            raise ValueError(f"Comprehensions not allowed in expression: {expr}")


def compile_safe_expression(expr, param_names, allowed_vars=None):
    """
    Validate and compile expression into a callable function.

    Args:
        expr: String expression to compile
        param_names: List of parameter names expected in expression
        allowed_vars: Set of allowed variable names (defaults to param_names + 'np')

    Returns:
        Compiled function that evaluates the expression

    Raises:
        ValueError: If expression validation fails
    """
    # Default allowed vars to param_names plus 'np'
    if allowed_vars is None:
        allowed_vars = set(param_names) | {"np"}

    # Normalize the expression
    normalized_expr = normalize_expression(expr)

    # Validate the expression
    validate_expression(normalized_expr, allowed_vars)

    # Compile the expression
    code = compile(normalized_expr, "<string>", "eval")

    # Create restricted namespace with only allowed functions
    safe_namespace = {
        "np": np,
        "__builtins__": {},  # Remove all builtins
    }
    safe_namespace.update(ALLOWED_FUNCTIONS)

    def compiled_func(*args):
        # Create local namespace with parameter values
        local_vars = dict(zip(param_names, args))
        return eval(code, safe_namespace, local_vars)

    return compiled_func

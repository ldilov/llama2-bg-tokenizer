from typing import Any, Dict, List, Union, get_origin, get_args


def validate_value(value: Any, expected_type: Any, path: str = '') -> bool:
    """
    Validate the type of the value against the expected type. Handles
    special typing constructs like Union, List, Dict, etc.
    """
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is Union:
        # For Union types, check if the value matches any of the options
        return any(validate_value(value, arg, path) for arg in args)
    elif origin in [list, List]:
        # For list types, check if all elements match the expected element type
        if not isinstance(value, list):
            raise TypeError(f"Expected a list at {path}, got {type(value).__name__}")
        for i, item in enumerate(value):
            validate_value(item, args[0], f"{path}[{i}]")
    elif origin in [dict, Dict]:
        # For dict types, check if keys and values match the expected types
        if not isinstance(value, dict):
            raise TypeError(f"Expected a dict at {path}, got {type(value).__name__}")
        for key, val in value.items():
            validate_value(key, args[0], f"{path}[{key}]")
            validate_value(val, args[1], f"{path}[{key}]")
    elif not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type} at {path}, got {type(value).__name__}")

    return True


def validate_config(config: Dict[str, Any], schema: Dict[str, Any], path: str = '') -> bool:
    """
    Validate a configuration dictionary against a schema.
    """
    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing key '{path}{key}' in config.")
        actual_value = config[key]
        validate_value(actual_value, expected_type, path + key)

    extra_keys = set(config.keys()) - set(schema.keys())
    if extra_keys:
        raise ValueError(f"Unexpected keys {extra_keys} in {path}.")

    return True
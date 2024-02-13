from typing import Any, Dict, List, Union, Type

ListOfType = lambda t: (list, t)

schema = {
    "add_bos_token": bool,
    "add_eos_token": bool,
    "added_tokens_decoder": Dict[str, dict],
    "additional_special_tokens": List[str],
    "special_tokens_attr": List[str],
    "dropout": float,
    "max_length": int,
    "min_frequency": int,
    "pad_to_multiple_of": Union[type(None), int],
    "bos_token": str,
    "eos_token": str,
    "unk_token": str,
    "fuse_unk": bool,
    "clean_up_tokenization_spaces": bool,
    "model_max_length": int,
    "pad_token": str,
    "pad_type_id": int,
    "padding_side": str,
    "sp_model_kwargs": dict,
    "spaces_between_special_tokens": bool,
    "tokenizer_class": str,
    "use_default_system_prompt": bool,
    "template": dict,
    "metadata": dict,
    "replacement_char": str
}
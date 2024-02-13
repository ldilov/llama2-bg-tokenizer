from transformers import LlamaTokenizer, LlamaTokenizerFast

TokenizerConfig = {
    "add_bos_token": True,
    "add_eos_token": True,
    "added_tokens_decoder": {
        "0": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "1": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "2": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    },
    "additional_special_tokens": [
        "<unk>",
        "<s>",
        "</s>"
    ],
    "special_tokens_attr": ["unk_token", "bos_token", "eos_token", "pad_token"],
    "dropout": 0.2,
    "max_length": 35,
    "min_frequency": 3,
    "pad_to_multiple_of": None,
    "bos_token": '<s>',
    "eos_token": '</s>',
    "unk_token": '<unk>',
    "fuse_unk": True,
    "clean_up_tokenization_spaces": False,
    "model_max_length": 255,
    "pad_token": '<s>',
    "pad_type_id": 0,
    "padding_side": "left",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": LlamaTokenizerFast.__name__,
    "use_default_system_prompt": False,
    "template": {
        "single": f"<s> $A",
        "pair": f"<s> $A <s> $B:1",
        "special_tokens": {
            "<s>": {
                "id": "<s>",
                "ids": [
                    1
                ],
                "tokens": [
                    "<s>"
                ]
            }
        }
    },
    "metadata": {
        "vocab_size": 32000
    },
    "replacement_char": "▁"
}
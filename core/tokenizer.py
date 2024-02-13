from transformers import PreTrainedTokenizerFast, SpecialTokensMixin, LlamaTokenizerFast


class ForwardingMeta(type):
    def __getattr__(cls, item):
        def method(*args, **kwargs):
            return getattr(PreTrainedTokenizerFast, item)(*args, **kwargs)

        return method


class GanioTokenizer(LlamaTokenizerFast, SpecialTokensMixin, metaclass=ForwardingMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
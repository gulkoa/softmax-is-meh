"""Integer-sequence tokenizer: space-separated ids in [0, 260); the
named specials <pad>=256 <bos>=257 <sep>=258 <eos>=259 are accepted as
readable aliases on encode; decoding yields the numeric form."""
import json
import os

from transformers import PreTrainedTokenizer

SPECIALS = {"<pad>": 256, "<bos>": 257, "<sep>": 258, "<eos>": 259}
VOCAB_SIZE = 260


class StieltjesTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "int_vocab.json"}

    def __init__(self, vocab_file=None, **kwargs):
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("sep_token", "<sep>")
        kwargs.setdefault("eos_token", "<eos>")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def get_vocab(self):
        vocab = {str(i): i for i in range(VOCAB_SIZE)}
        vocab.update(SPECIALS)   # readable aliases share the numeric ids
        return vocab

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        if token in SPECIALS:
            return SPECIALS[token]
        try:
            i = int(token)
        except ValueError:
            return SPECIALS["<pad>"]
        return i if 0 <= i < VOCAB_SIZE else SPECIALS["<pad>"]

    def _convert_id_to_token(self, index):
        return str(index)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        name = (filename_prefix + "-" if filename_prefix else "") + "int_vocab.json"
        path = os.path.join(save_directory, name)
        with open(path, "w") as f:
            json.dump(self.get_vocab(), f)
        return (path,)

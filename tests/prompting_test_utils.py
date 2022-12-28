from typing import Union

import torch

KNOWN_WORDS = ['a', 'b', 'c']
KNOWN_WORDS_TOKEN_IDS = [0, 1, 2]
UNKNOWN_WORDS = ['d', 'e', 'f']

class DummyEmbeddingsList(list):
    def __getattr__(self, name):
        if name == 'num_embeddings':
            return len(self)
        elif name == 'weight':
            return self
        elif name == 'data':
            return self

class DummyTransformer:
    def __init__(self):
        self.embeddings = DummyEmbeddingsList([0] * len(KNOWN_WORDS))

    def resize_token_embeddings(self, new_size=None):
        if new_size is None:
            return self.embeddings
        else:
            while len(self.embeddings) > new_size:
                self.embeddings.pop(-1)
            while len(self.embeddings) < new_size:
                self.embeddings.append(0)

    def get_input_embeddings(self):
        return self.embeddings

class DummyTokenizer():
    def __init__(self):
        self.tokens = KNOWN_WORDS.copy()
        self.bos_token_id = 49406 # these are what the real CLIPTokenizer has
        self.eos_token_id = 49407
        self.pad_token_id = 49407
        self.unk_token_id = 49407

    def convert_tokens_to_ids(self, token_str):
        try:
            return self.tokens.index(token_str)
        except ValueError:
            return self.unk_token_id

    def add_tokens(self, token_str):
        if token_str in self.tokens:
            return 0
        self.tokens.append(token_str)
        return 1


class DummyClipEmbedder:
    def __init__(self):
        self.max_length = 77
        self.transformer = DummyTransformer()
        self.tokenizer = DummyTokenizer()
        self.position_embeddings_tensor = torch.randn([77,768], dtype=torch.float32)

    def position_embedding(self, indices: Union[list,torch.Tensor]):
        if type(indices) is list:
            indices = torch.tensor(indices, dtype=int)
        return torch.index_select(self.position_embeddings_tensor, 0, indices)


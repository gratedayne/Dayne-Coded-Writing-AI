class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(text.split())
        tokens = sorted(tokens)
        self.vocab = {tok: i for i, tok in enumerate(tokens, start=1)}  # reserve 0 for padding
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(tok, 0) for tok in text.split()]

    def decode(self, token_ids):
        return " ".join(self.inv_vocab.get(i, "[UNK]") for i in token_ids)

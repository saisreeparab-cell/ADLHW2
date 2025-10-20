import abc

import torch



def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        """
        Init AutoregressiveModel class
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        self.token_emb = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_emb = None
        self.bos = torch.nn.Parameter(torch.zeros(1, 1, d_latent))

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = d_latent,
            nhead = 8,
            dim_feedforward = 4* d_latent,
            dropout=0,
            batch_first=True,
            activation="relu",
            norm_first=True
        )

        self.encoder =  torch.nn.TransformerEncoder(encoder_layer, num_layers = 4)
        self.head = torch.nn.Linear(d_latent, n_tokens)
        # raise NotImplementedError()

    @staticmethod
    def _causal_mask(seq_len: int, device):
        return torch.triu(torch.ones(seq_len, seq_len, device = device, dtype = torch.bool), diagonal = 1)

    def _ensure_pos_emb(self,seq_len: int, device):
        if self.pos_emb is None or self.pos_emb.num_embeddings < seq_len:
            self.pos_emb = torch.nn.Embedding(seq_len, self.d_latent).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
            Return the forward method based on parameters. 
        """
        b, h, w = x.shape
        l = h * w
        device = x.device
        seq = x.view(b, l)
        tok = self.token_emb(seq)
        bos = self.bos.expand(b, 1, self.d_latent)
        x_in = torch.cat([bos, tok[:, :-1, :]], dim = 1)

        self._ensure_pos_emb(l, device)
        pos_ids = torch.arange(l, device=device).unsqueeze(0)
        x_in = x_in + self.pos_emb(pos_ids)
        mask = self._causal_mask(l, device)
        hidden = self.encoder(x_in, mask = mask)
        logits = self.head(hidden).view( b, h, w, self.n_tokens)
        return logits, {}


        # raise NotImplementedError()

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
            Return the generate method based on parameters. 
        """

        device = device or next(self.parameters()).device
        l = h * w
        d = self.d_latent

        seq = torch.zeros(B, 0, dtype= torch.long, device=device)

        for t in range(l):
            if t ==0:
                x_in = self.bos.expand(B, 1, d)
            else:
                tok = self.token_emb(seq)
                x_in = torch.cat([self.bos.expand(B, 1, d), tok], dim = 1)

            self._ensure_pos_emb(l, device)
            pos = torch.arange(t + 1, device = device).unsqueeze(0)
            x_in = x_in + self.pos_emb(pos)

            mask = self._causal_mask(t + 1, device)
            hidden = self.encoder(x_in, mask = mask)
            logits = self.head(hidden[:, -1, :])
            next_tok = torch.argmax(logits, dim = 1).squeeze(1)
            seq = torch.cat([seq, next_tok], dim =1)

        return seq.view(B, h, w)


        # raise NotImplementedError()

"""
Volby-0.1
Karpathy-style character-level GPT training
Volbasty Studios
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/input.txt"

BATCH_SIZE = 64
BLOCK_SIZE = 128

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

LEARNING_RATE = 3e-4

N_EMBD = 256
N_HEAD = 4
N_LAYER = 4

DROPOUT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


# ============================================================
# LOAD DATA
# ============================================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

chars = sorted(list(set(text)))

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [
        stoi[c]
        for c in s
    ]


def decode(l):
    return "".join(
        itos[i]
        for i in l
    )


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

n = int(
    0.9 * len(data)
)

train_data = data[:n]

val_data = data[n:]

print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(val_data)
)


# ============================================================
# BATCH CREATION
# ============================================================

def get_batch(split):

    source = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        len(source) - BLOCK_SIZE - 1,
        (
            BATCH_SIZE,
        )
    )

    x = torch.stack(
        [
            source[
                i:
                i + BLOCK_SIZE
            ]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            source[
                i + 1:
                i + BLOCK_SIZE + 1
            ]
            for i in ix
        ]
    )

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# LOSS ESTIMATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in [
        "train",
        "val"
    ]:

        losses = torch.zeros(
            EVAL_ITERS
        )

        for k in range(
            EVAL_ITERS
        ):

            X, Y = get_batch(
                split
            )

            _, loss = model(
                X,
                Y
            )

            losses[k] = (
                loss.item()
            )

        results[split] = (
            losses.mean()
        )

    model.train()

    return results


# ============================================================
# MODEL COMPONENTS
# ============================================================

class Head(nn.Module):

    def __init__(
        self,
        head_size
    ):

        super().__init__()

        self.key = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )


    def forward(
        self,
        x
    ):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        wei = (
            q
            @ k.transpose(
                -2,
                -1
            )
        )

        wei = wei * (
            k.shape[-1]
            ** -0.5
        )

        wei = wei.masked_fill(
            self.tril[
                :T,
                :T
            ] == 0,
            float(
                "-inf"
            )
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        wei = self.dropout(
            wei
        )

        v = self.value(x)

        out = wei @ v

        return out


# ============================================================
# MULTI HEAD ATTENTION
# ============================================================

class MultiHeadAttention(
    nn.Module
):

    def __init__(
        self,
        num_heads,
        head_size
    ):

        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(
                    head_size
                )
                for _ in range(
                    num_heads
                )
            ]
        )

        self.proj = nn.Linear(
            N_EMBD,
            N_EMBD
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )


    def forward(
        self,
        x
    ):

        out = torch.cat(
            [
                h(x)
                for h in self.heads
            ],
            dim=-1
        )

        out = self.proj(
            out
        )

        out = self.dropout(
            out
        )

        return out


# ============================================================
# FEED FORWARD
# ============================================================

class FeedForward(
    nn.Module
):

    def __init__(
        self,
        n_embd
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.ReLU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            ),

            nn.Dropout(
                DROPOUT
            )

        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(
    nn.Module
):

    def __init__(
        self,
        n_embd,
        n_head
    ):

        super().__init__()

        head_size = (
            n_embd
            // n_head
        )

        self.sa = (
            MultiHeadAttention(
                n_head,
                head_size
            )
        )

        self.ffwd = (
            FeedForward(
                n_embd
            )
        )

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )


    def forward(
        self,
        x
    ):

        x = (
            x
            + self.sa(
                self.ln1(x)
            )
        )

        x = (
            x
            + self.ffwd(
                self.ln2(x)
            )
        )

        return x


# ============================================================
# GPT MODEL
# ============================================================

class GPTLanguageModel(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.token_embedding_table = (
            nn.Embedding(
                vocab_size,
                N_EMBD
            )
        )

        self.position_embedding_table = (
            nn.Embedding(
                BLOCK_SIZE,
                N_EMBD
            )
        )

        self.blocks = nn.Sequential(

            *[
                Block(
                    N_EMBD,
                    N_HEAD
                )
                for _ in range(
                    N_LAYER
                )
            ]

        )

        self.ln_f = nn.LayerNorm(
            N_EMBD
        )

        self.lm_head = nn.Linear(
            N_EMBD,
            vocab_size
        )


    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        tok_emb = (
            self.token_embedding_table(
                idx
            )
        )

        pos_emb = (
            self.position_embedding_table(
                torch.arange(
                    T,
                    device=DEVICE
                )
            )
        )

        x = (
            tok_emb
            + pos_emb
        )

        x = self.blocks(
            x
        )

        x = self.ln_f(
            x
        )

        logits = self.lm_head(
            x
        )

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return (
            logits,
            loss
        )


    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(
            max_new_tokens
        ):

            idx_cond = (
                idx[
                    :,
                    -BLOCK_SIZE:
                ]
            )

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :,
                -1,
                :
            ]

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (
                    idx,
                    idx_next
                ),
                dim=1
            )

        return idx


# ============================================================
# CREATE MODEL
# ============================================================

model = GPTLanguageModel()

model = model.to(
    DEVICE
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print(
    "\nStarting training..."
)

for iteration in range(
    MAX_ITERS
):

    if (
        iteration
        % EVAL_INTERVAL
        == 0
    ):

        losses = estimate_loss()

        print(
            f"Step {iteration}: "
            f"train {losses['train']:.4f}, "
            f"val {losses['val']:.4f}"
        )


    X, Y = get_batch(
        "train"
    )

    logits, loss = model(
        X,
        Y
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "stoi":
        stoi,

    "itos":
        itos,

    "vocab_size":
        vocab_size,

    "block_size":
        BLOCK_SIZE,

    "n_embd":
        N_EMBD,

    "n_head":
        N_HEAD,

    "n_layer":
        N_LAYER

}


torch.save(

    checkpoint,

    "checkpoints/volby-0.1.pth"

)


print(
    "\nTraining complete."
)

print(
    "Saved:"
)

print(
    "checkpoints/volby-0.1.pth"
)
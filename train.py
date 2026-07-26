"""
Volby-0.2
Raw Text Pretraining Pipeline
Developed by Volbasty Studios
"""

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# SETTINGS
# ============================================================

PRETRAIN_PATH = "data/pretrain.txt"
VALIDATION_PATH = "data/validation.txt"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "volby-0.2-best.pth"
)

FINAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "volby-0.2-final.pth"
)

BATCH_SIZE = 64
BLOCK_SIZE = 128

MAX_ITERS = 5000

EVAL_INTERVAL = 250
EVAL_ITERS = 100

LEARNING_RATE = 3e-4

N_EMBD = 256
N_HEAD = 4
N_LAYER = 4

DROPOUT = 0.2

PATIENCE = 5

GRAD_CLIP = 1.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(1337)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


print("=" * 40)
print("       VOLBY-0.2 PRETRAINING")
print("       Volbasty Studios")
print("=" * 40)

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD DATA
# ============================================================

with open(
    PRETRAIN_PATH,
    "r",
    encoding="utf-8"
) as f:

    train_text = f.read()


with open(
    VALIDATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    val_text = f.read()


print(
    "Training characters:",
    len(train_text)
)

print(
    "Validation characters:",
    len(val_text)
)


# ============================================================
# CREATE VOCABULARY
# ============================================================

chars = sorted(
    list(
        set(
            train_text
            + val_text
        )
    )
)

vocab_size = len(
    chars
)

print(
    "Vocabulary size:",
    vocab_size
)


stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(text):

    return [
        stoi[c]
        for c in text
    ]


def decode(ids):

    return "".join(
        itos[i]
        for i in ids
    )


# ============================================================
# TOKENIZE DATA
# ============================================================

train_data = torch.tensor(
    encode(train_text),
    dtype=torch.long
)

val_data = torch.tensor(
    encode(val_text),
    dtype=torch.long
)


print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(val_data)
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

if len(train_data) <= BLOCK_SIZE:

    raise ValueError(
        "Training data is too small "
        "for the selected BLOCK_SIZE."
    )


if len(val_data) <= BLOCK_SIZE:

    raise ValueError(
        "Validation data is too small "
        "for the selected BLOCK_SIZE."
    )


# ============================================================
# BATCH CREATION
# ============================================================

def get_batch(
    split
):

    data = (
        train_data
        if split == "train"
        else val_data
    )

    ix = torch.randint(
        0,
        len(data)
        - BLOCK_SIZE
        - 1,
        (
            BATCH_SIZE,
        )
    )

    x = torch.stack(
        [
            data[
                i:
                i + BLOCK_SIZE
            ]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[
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
# ATTENTION HEAD
# ============================================================

class Head(
    nn.Module
):

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

        return wei @ v


# ============================================================
# MULTI-HEAD ATTENTION
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
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        out = self.proj(
            out
        )

        return self.dropout(
            out
        )


# ============================================================
# FEED FORWARD NETWORK
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

            nn.GELU(),

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

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
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


    def forward(
        self,
        x
    ):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
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

        token_embeddings = (
            self.token_embedding_table(
                idx
            )
        )

        position_embeddings = (
            self.position_embedding_table(
                torch.arange(
                    T,
                    device=idx.device
                )
            )
        )

        x = (
            token_embeddings
            + position_embeddings
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

            logits = logits.reshape(
                B * T,
                C
            )

            targets = targets.reshape(
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
                [
                    idx,
                    idx_next
                ],
                dim=1
            )

        return idx


# ============================================================
# LOSS EVALUATION
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
            losses.mean().item()
        )

    model.train()

    return results


# ============================================================
# CREATE MODEL
# ============================================================

model = GPTLanguageModel()

model = model.to(
    DEVICE
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Model parameters:",
    parameter_count
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=0.01
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=MAX_ITERS
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_loss = float(
    "inf"
)

patience_counter = 0


print(
    "\nStarting training..."
)


# ============================================================
# TRAINING LOOP
# ============================================================

for iteration in range(
    MAX_ITERS
):

    if (
        iteration
        % EVAL_INTERVAL
        == 0
    ):

        losses = estimate_loss()

        train_loss = (
            losses["train"]
        )

        val_loss = (
            losses["val"]
        )

        perplexity = math.exp(
            min(
                val_loss,
                20
            )
        )

        print(
            "\n"
            + "=" * 40
        )

        print(
            f"Step {iteration}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Perplexity: "
            f"{perplexity:.4f}"
        )

        print(
            "=" * 40
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            patience_counter = 0

            torch.save(

                {
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

                },

                BEST_MODEL_PATH

            )

            print(
                "New best model saved."
            )

        else:

            patience_counter += 1

            print(
                f"No improvement."
            )

            print(
                f"Patience: "
                f"{patience_counter}/"
                f"{PATIENCE}"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


    # ========================================================
    # TRAIN STEP
    # ========================================================

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


    # ========================================================
    # GRADIENT CLIPPING
    # ========================================================

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )


    optimizer.step()

    scheduler.step()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)

best_checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(

    {

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

    },

    FINAL_MODEL_PATH

)


print(
    "\n"
    + "=" * 40
)

print(
    "       VOLBY-0.2 COMPLETE"
)

print(
    "       Volbasty Studios"
)

print(
    "=" * 40
)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Final model:",
    FINAL_MODEL_PATH
)

print(
    "Best model:",
    BEST_MODEL_PATH
)
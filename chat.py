import torch
import torch.nn.functional as F

from train import GPTLanguageModel


MODEL_PATH = "checkpoints/volby-0.1.pth"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 300

TEMPERATURE = 0.8


# =========================
# Load checkpoint
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

BLOCK_SIZE = checkpoint["block_size"]


# =========================
# Create model
# =========================

model = GPTLanguageModel()

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    DEVICE
)

model.eval()


# =========================
# Encode
# =========================

def encode(text):

    return [
        stoi[c]
        for c in text
        if c in stoi
    ]


# =========================
# Decode
# =========================

def decode(ids):

    return "".join(
        itos[i]
        for i in ids
    )


# =========================
# Generate
# =========================

@torch.no_grad()
def generate(prompt):

    ids = encode(
        prompt
    )

    if not ids:

        ids = [
            0
        ]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(
        MAX_NEW_TOKENS
    ):

        x_cond = x[
            :,
            -BLOCK_SIZE:
        ]

        logits, _ = model(
            x_cond
        )

        logits = logits[
            :,
            -1,
            :
        ]

        logits = (
            logits
            / TEMPERATURE
        )

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            1
        )

        x = torch.cat(
            [
                x,
                next_token
            ],
            dim=1
        )

    return decode(
        x[0].tolist()
    )


# =========================
# Chat
# =========================

print(
    "================================"
)

print(
    "          VOLBY-0.1"
)

print(
    "       Volbasty Studios"
)

print(
    "================================"
)

print(
    "Device:",
    DEVICE
)

print(
    "\nVolby is ready!"
)

print(
    "Type 'exit' to quit."
)


while True:

    prompt = input(
        "\nYou: "
    )

    if prompt.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "Volby: Goodbye!"
        )

        break

    print(
        "\nVolby:",
        generate(
            prompt
        )
    )
import torch
import torch.nn.functional as F

from train import GPTLanguageModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "checkpoints/volby-0.2-best.pth"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 300

TEMPERATURE = 0.8


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

stoi = checkpoint["stoi"]

itos = checkpoint["itos"]

vocab_size = checkpoint["vocab_size"]

BLOCK_SIZE = checkpoint["block_size"]


# ============================================================
# CREATE MODEL
# ============================================================

model = GPTLanguageModel(
    vocab_size
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    DEVICE
)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# ENCODE
# ============================================================

def encode(
    text
):

    return [
        stoi[c]
        for c in text
        if c in stoi
    ]


# ============================================================
# DECODE
# ============================================================

def decode(
    ids
):

    return "".join(
        itos[i]
        for i in ids
    )


# ============================================================
# GENERATE
# ============================================================

@torch.no_grad()
def generate(
    prompt
):

    prompt_ids = encode(
        prompt
    )

    if not prompt_ids:

        prompt_ids = [0]


    x = torch.tensor(

        [prompt_ids],

        dtype=torch.long,

        device=DEVICE

    )


    prompt_length = (
        x.shape[1]
    )


    for _ in range(
        MAX_NEW_TOKENS
    ):

        x_cond = (

            x[
                :,
                -BLOCK_SIZE:
            ]

        )


        logits, _ = model(
            x_cond
        )


        logits = (

            logits[
                :,
                -1,
                :
            ]

        )


        logits = (

            logits
            / TEMPERATURE

        )


        probabilities = (

            F.softmax(

                logits,

                dim=-1

            )

        )


        next_token = (

            torch.multinomial(

                probabilities,

                num_samples=1

            )

        )


        x = torch.cat(

            [

                x,

                next_token

            ],

            dim=1

        )


    generated_ids = (

        x[
            0,
            prompt_length:
        ]

        .tolist()

    )


    return decode(
        generated_ids
    )


# ============================================================
# CHAT
# ============================================================

print(
    "================================"
)

print(
    "          VOLBY-0.2"
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
    "Model:",
    MODEL_PATH
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


    response = generate(
        prompt
    )


    print(
        "\nVolby:",
        response
    )
"""Synthetic deep-recall conversations (nope-it v2 training slice).

Each conversation: a fact stated in an early user turn, filler dialogue
to a target token depth, then a probe turn whose answer REQUIRES the
fact — explicit training demand for deep back-reference (the untrimmed
v1 SFT showed architecture alone buys nothing: finding
2026-08-06-nopeit-v1-no-deep-recall).

Facts/fillers are templated with disjoint vocab from the eval battery
(eval_deepctx_recall_stilt) — the eval's exact facts never appear, so
the gauntlet stays held-out.

Usage: python gen_deeprecall_data.py [--n 8000] [--out results/deeprecall_convos.json]
"""
import argparse
import json
import random

from transformers import AutoTokenizer

NAMES = ["Maya", "Omar", "Lena", "Kai", "Priya", "Tom", "Zoe", "Ivan",
         "Nadia", "Leo", "Sara", "Finn"]
COLORS = ["green", "purple", "orange", "teal", "crimson", "silver"]
CITIES = ["Oslo", "Lima", "Cairo", "Seoul", "Porto", "Quebec"]
PETS = [("cat", "Milo"), ("parrot", "Kiwi"), ("hamster", "Peanut"),
        ("turtle", "Sheldon"), ("goldfish", "Bubbles")]
WORDS = ["lantern", "cactus", "meteor", "violin", "harbor", "tulip"]
NUMBERS = ["23", "42", "7", "88", "64", "31"]

FILLERS = [
    ("Can you explain how tides work?",
     "Tides are caused mainly by the Moon's gravity pulling on Earth's "
     "oceans, creating bulges of water that move as Earth rotates."),
    ("What makes sourdough bread different?",
     "Sourdough uses a wild yeast starter instead of commercial yeast, "
     "which gives it a tangy flavor and chewier texture."),
    ("Tell me something interesting about octopuses.",
     "Octopuses have three hearts and blue blood, and each arm has its "
     "own cluster of neurons that can act semi-independently."),
    ("How do noise-canceling headphones work?",
     "They use microphones to detect ambient sound and play an inverted "
     "sound wave that cancels much of the noise before you hear it."),
    ("Why do leaves change color in autumn?",
     "Chlorophyll breaks down as days shorten, revealing yellow and "
     "orange pigments that were in the leaf all along."),
    ("What's a good way to start learning chess?",
     "Learn how each piece moves, practice simple checkmates, and play "
     "short games focusing on controlling the center."),
    ("How are rainbows formed?",
     "Sunlight refracts and reflects inside raindrops, splitting into "
     "its component colors and forming an arc opposite the sun."),
    ("What does a compiler do?",
     "A compiler translates source code written in a programming "
     "language into machine code the processor can execute."),
    ("Why is the sky blue?",
     "Air molecules scatter short blue wavelengths of sunlight more "
     "than red ones, so scattered blue light reaches your eyes from "
     "every direction."),
    ("Give me a tip for better sleep.",
     "Keep a consistent bedtime and avoid bright screens for the last "
     "hour before sleep so your body can wind down."),
]


def make_convo(rng, tok, depth):
    kind = rng.randrange(5)
    if kind == 0:
        n = rng.choice(NAMES)
        fact = f"By the way, my sister's name is {n}."
        probe = "What is my sister's name?"
        answer = f"Your sister's name is {n}."
    elif kind == 1:
        c = rng.choice(COLORS)
        fact = f"I just painted my bike {c}."
        probe = "What color is my bike now?"
        answer = f"Your bike is {c}."
    elif kind == 2:
        city = rng.choice(CITIES)
        fact = f"I'm planning a trip to {city} next month."
        probe = "Which city am I visiting next month?"
        answer = f"You're visiting {city}."
    elif kind == 3:
        pet, pname = rng.choice(PETS)
        fact = f"My {pet} is called {pname}."
        probe = f"What did I say my {pet} is called?"
        answer = f"Your {pet} is called {pname}."
    else:
        w, num = rng.choice(WORDS), rng.choice(NUMBERS)
        fact = f"Remember this: the passphrase is {w} {num}."
        probe = "What was the passphrase?"
        answer = f"The passphrase is {w} {num}."

    msgs = [{"role": "user", "content": fact},
            {"role": "assistant", "content": "Got it, I'll remember that."}]
    count = len(tok(fact + answer, add_special_tokens=False).input_ids) + 20
    order = rng.sample(range(len(FILLERS)), len(FILLERS))
    i = 0
    while count < depth:
        q, a = FILLERS[order[i % len(FILLERS)]]
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
        count += len(tok(q + a, add_special_tokens=False).input_ids) + 8
        i += 1
    msgs.append({"role": "user", "content": probe})
    msgs.append({"role": "assistant", "content": answer})
    return {"messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--out", default="/users/PAS2402/alexg/softmax/"
                    "softmax-is-meh/results/deeprecall_convos.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained("gpt2")
    convos = []
    for i in range(args.n):
        # bias toward genuinely deep placements; cap under train ctx 2048
        depth = rng.choice([300, 600, 900, 1200, 1500, 1800,
                            1200, 1500, 1800])
        convos.append(make_convo(rng, tok, depth))
    json.dump(convos, open(args.out, "w"))
    print(f"wrote {len(convos)} deep-recall convos -> {args.out}")


if __name__ == "__main__":
    main()

"""Synthetic logic/reasoning tasks for the reasoning-RL line (plan:
thesis/findings/2026-07-22-plan-judge-rl-chat.md): verifiable by
construction, difficulty-tunable — the curriculum floor analogous to
gen_synth_code_tasks for the code line.

Templates (difficulty 1-3 scales chain length / ranges):
  arithmetic chains, predicate counting, order deduction,
  sequence next-term, letter counting.

Row schema: {task_id, template, difficulty, question, answer,
aliases} — answer checked by normalized exact match (numbers get
int-string aliases).

Usage: python gen_logic_tasks.py [--n 600] [--difficulty 1,2,3]
"""
import argparse
import json
import random

NAMES = ["Ava", "Ben", "Cleo", "Dan", "Eve", "Finn", "Gia", "Hugo"]
ITEMS = ["apples", "books", "coins", "pens", "shells", "stamps"]
WORDS = ["banana", "committee", "parallel", "mississippi", "letter",
         "balloon", "coffee", "success", "bubble", "pepper"]


def arith_chain(rng, d):
    n_ops = {1: 2, 2: 3, 3: 5}[d]
    hi = {1: 12, 2: 30, 3: 60}[d]
    val = rng.randint(2, hi)
    text = f"Start with {val}."
    for _ in range(n_ops):
        op = rng.choice(["add", "subtract", "multiply"])
        if op == "multiply" and val < 200:
            k = rng.randint(2, 4)
            text += f" Multiply by {k}."
            val *= k
        elif op == "add":
            k = rng.randint(2, hi)
            text += f" Add {k}."
            val += k
        else:
            k = rng.randint(2, min(hi, max(2, val - 1)))
            text += f" Subtract {k}."
            val -= k
    return (text + " What is the result?", str(val))


def count_predicate(rng, d):
    n = {1: 6, 2: 10, 3: 16}[d]
    xs = [rng.randint(1, 50) for _ in range(n)]
    kind = rng.choice(["even", "odd", "greater than 20"])
    if kind == "even":
        ans = sum(1 for x in xs if x % 2 == 0)
    elif kind == "odd":
        ans = sum(1 for x in xs if x % 2 == 1)
    else:
        ans = sum(1 for x in xs if x > 20)
    return (f"How many numbers in this list are {kind}? "
            f"{', '.join(map(str, xs))}", str(ans))


def order_deduction(rng, d):
    k = {1: 3, 2: 4, 3: 5}[d]
    ppl = rng.sample(NAMES, k)
    order = ppl[:]
    rng.shuffle(order)                    # order[0] tallest
    facts = [f"{order[i]} is taller than {order[i + 1]}."
             for i in range(k - 1)]
    rng.shuffle(facts)
    which = rng.choice(["tallest", "shortest"])
    ans = order[0] if which == "tallest" else order[-1]
    return (" ".join(facts) + f" Who is the {which}?", ans)


def sequence_next(rng, d):
    if rng.random() < 0.5:
        a = rng.randint(1, 12)
        step = rng.randint(2, {1: 6, 2: 12, 3: 25}[d])
        seq = [a + i * step for i in range(4)]
        nxt = a + 4 * step
    else:
        a = rng.randint(1, 5)
        r = rng.choice([2, 3] if d < 3 else [2, 3, 4])
        seq = [a * r ** i for i in range(4)]
        nxt = a * r ** 4
    return (f"What number comes next in the sequence: "
            f"{', '.join(map(str, seq))}, ...?", str(nxt))


def letter_count(rng, d):
    w = rng.choice(WORDS)
    ch = rng.choice(sorted(set(w)))
    return (f"How many times does the letter '{ch}' appear in the "
            f"word '{w}'?", str(w.count(ch)))


def share_items(rng, d):
    a, b = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    hi = {1: 20, 2: 60, 3: 150}[d]
    n1 = rng.randint(2, hi)
    n2 = rng.randint(2, hi)
    give = rng.randint(1, n1)
    q = (f"{a} has {n1} {item} and {b} has {n2} {item}. {a} gives "
         f"{give} {item} to {b}. How many {item} does {b} have now?")
    return (q, str(n2 + give))


TEMPLATES = [arith_chain, count_predicate, order_deduction,
             sequence_next, letter_count, share_items]


def make_task(rng, idx, d):
    fn = rng.choice(TEMPLATES)
    q, ans = fn(rng, d)
    aliases = [ans]
    if ans.lstrip("-").isdigit():
        aliases += [f"{int(ans):,}"]
    return {"task_id": f"logic-{idx:05d}", "template": fn.__name__,
            "difficulty": d, "question": q, "answer": ans,
            "aliases": aliases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--difficulty", default="1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logic_tasks_v1.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    diffs = [int(x) for x in args.difficulty.split(",")]
    tasks = [make_task(rng, i, diffs[i % len(diffs)])
             for i in range(args.n)]
    json.dump(tasks, open(args.out, "w"), indent=1)
    mix = {}
    for t in tasks:
        key = f"{t['template']}/d{t['difficulty']}"
        mix[key] = mix.get(key, 0) + 1
    print(f"wrote {len(tasks)} tasks -> {args.out}")
    print("mix:", dict(sorted(mix.items())))


if __name__ == "__main__":
    main()

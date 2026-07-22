"""Synthetic easy-task generator for code RL (plan: thesis/findings/
2026-07-20-plan-rl-coding-stilt11.md): unlimited MBPP-schema one-liner
problems (string/list/math), tests computed by executing the reference
solution — verifiable by construction. Output rows carry the same
fields the GRPO loader uses (text, test_list, task_id) so they mix
directly into the curriculum.

Usage: python gen_synth_code_tasks.py [--n 500] [--seed 0]
                                      [--out synth_code_tasks.json]
"""
import argparse
import json
import random

WORDS = ["apple", "banana", "kiwi", "mango", "pear", "plum", "grape",
         "melon", "peach", "berry", "lemon", "olive"]


def sample_list(rng, lo=-20, hi=40, n=(3, 7)):
    return [rng.randint(lo, hi) for _ in range(rng.randint(*n))]


TEMPLATES = [
    dict(name="reverse_string",
         text="Write a function reverse_string(s) that returns the "
              "string s reversed.",
         sol=lambda: (lambda s: s[::-1]),
         args=lambda rng: [(rng.choice(WORDS),) for _ in range(3)]),
    dict(name="count_char",
         text="Write a function count_char(s, c) that returns how many "
              "times character c appears in string s.",
         sol=lambda: (lambda s, c: s.count(c)),
         args=lambda rng: [(w := rng.choice(WORDS), rng.choice(w))
                           for _ in range(3)]),
    dict(name="sum_list",
         text="Write a function sum_list(xs) that returns the sum of a "
              "list of integers xs.",
         sol=lambda: (lambda xs: sum(xs)),
         args=lambda rng: [(sample_list(rng),) for _ in range(3)]),
    dict(name="max_of_list",
         text="Write a function max_of_list(xs) that returns the "
              "largest element of a non-empty list xs.",
         sol=lambda: (lambda xs: max(xs)),
         args=lambda rng: [(sample_list(rng),) for _ in range(3)]),
    dict(name="filter_even",
         text="Write a function filter_even(xs) that returns a list of "
              "only the even numbers in xs, in order.",
         sol=lambda: (lambda xs: [x for x in xs if x % 2 == 0]),
         args=lambda rng: [(sample_list(rng),) for _ in range(3)]),
    dict(name="square_all",
         text="Write a function square_all(xs) that returns a list with "
              "every element of xs squared.",
         sol=lambda: (lambda xs: [x * x for x in xs]),
         args=lambda rng: [(sample_list(rng, -9, 9),) for _ in range(3)]),
    dict(name="is_palindrome",
         text="Write a function is_palindrome(s) that returns True if "
              "the string s reads the same forwards and backwards.",
         sol=lambda: (lambda s: s == s[::-1]),
         args=lambda rng: [(rng.choice([w, w + w[::-1]]),)
                           for w in [rng.choice(WORDS) for _ in range(3)]]),
    dict(name="factorial",
         text="Write a function factorial(n) that returns n! for a "
              "non-negative integer n.",
         sol=lambda: (lambda n: 1 if n < 2 else
                      __import__("math").factorial(n)),
         args=lambda rng: [(rng.randint(0, 8),) for _ in range(3)]),
    dict(name="fizzbuzz_value",
         text="Write a function fizzbuzz_value(n) that returns 'Fizz' "
              "if n is divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by "
              "both, else n itself.",
         sol=lambda: (lambda n: "FizzBuzz" if n % 15 == 0 else
                      "Fizz" if n % 3 == 0 else
                      "Buzz" if n % 5 == 0 else n),
         args=lambda rng: [(rng.randint(1, 60),) for _ in range(3)]),
    dict(name="vowel_count",
         text="Write a function vowel_count(s) that returns the number "
              "of vowels (aeiou) in the lowercase string s.",
         sol=lambda: (lambda s: sum(ch in "aeiou" for ch in s)),
         args=lambda rng: [(rng.choice(WORDS) + rng.choice(WORDS),)
                           for _ in range(3)]),
]


def make_task(rng, idx):
    t = rng.choice(TEMPLATES)
    fn = t["sol"]()
    cases = t["args"](rng)
    tests = [f"assert {t['name']}({', '.join(map(repr, c))}) == "
             f"{fn(*c)!r}" for c in cases]
    return {"task_id": f"synth-{idx:05d}", "source": "synthetic",
            "template": t["name"], "text": t["text"],
            "test_list": tests, "test_setup_code": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synth_code_tasks.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tasks = [make_task(rng, i) for i in range(args.n)]
    json.dump(tasks, open(args.out, "w"), indent=1)
    tmpl_counts = {}
    for t in tasks:
        tmpl_counts[t["template"]] = tmpl_counts.get(t["template"], 0) + 1
    print(f"wrote {len(tasks)} tasks -> {args.out}")
    print("template mix:", tmpl_counts)


if __name__ == "__main__":
    main()

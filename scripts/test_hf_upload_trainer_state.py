"""Selection and naming of the trainer-state backup (no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_upload_trainer_state as up  # noqa: E402

NAMES = [
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt",
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt.bak",            # excluded: identical copy
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.step1000.pt",
    "ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s3.step9000.pt",
    "ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s3.pt",                 # excluded: final, on the Hub as a model
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s2.step1000.pt",        # excluded: not a 1e-3 contrast run
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s0.step1000.pt",        # excluded: seed 0
]


def test_select_and_labels(tmp_path):
    for n in NAMES:
        (tmp_path / n).write_bytes(b"x")
    got = up.select(str(tmp_path))
    repo_paths = [r for _, r in got]
    assert repo_paths == [
        "stj-q4-nope-lrsweep-lr0.001-s2/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt",
        "stj-q4-nope-lrsweep-lr0.001-s2/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.step1000.pt",
        "sdpa-nope-lrsweepls-lr0.001-s3/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s3.step9000.pt",
    ]
    assert all(os.path.isfile(p) for p, _ in got)
    assert up.step_of("a.step1000.pt") == 1000 and up.step_of("a.pt") is None
    assert up.run_label("ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.step1000.pt") == "stj-q4-nope-lrsweep-lr0.001-s2"


def test_card_lists_every_file(tmp_path):
    for n in NAMES[:1] + NAMES[2:4]:
        (tmp_path / n).write_bytes(b"xy")
    files = up.select(str(tmp_path))
    steps = {r: up.step_of(os.path.basename(r)) or 4500 for _, r in files}
    text = up.card(files, steps)
    assert text.startswith("---\nlicense: mit")
    for _, r in files:
        assert f"`{r}`" in text
    assert "| 4,500 |" in text and "| 9,000 |" in text

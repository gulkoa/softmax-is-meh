#!/usr/bin/env python3
"""Tests for scripts/score_length_alpha_grid.py on synthetic logs (CPU, no data).

    python scripts/test_score_length_alpha_grid.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_length_alpha_grid import best, gate, parse_blocks, union, verdict  # noqa: E402


def write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".out", delete=False)
    f.write(text)
    f.close()
    return f.name


TABLE = ("nope=True  alpha sweep=[0.25, 0.3]\n"
         "      L        off     a=0.25      a=0.3\n"
         "   1024       27.6       27.6       27.6\n"
         "   2048      313.8       29.0       28.6\n"
         "   4096     1218.7      141.6       56.1\n"
         "   8192     1274.0      307.5      230.0\n"
         "DONE\n")


def test_parse_named_and_unnamed_blocks():
    named = parse_blocks(write("x step 1\n=== a.pt\n" + TABLE + "=== b.pt\n" + TABLE + "ALL DONE\n"))
    assert set(named) == {"a.pt", "b.pt"}
    assert named["a.pt"][8192] == {"off": 1274.0, 0.25: 307.5, 0.3: 230.0}
    single = parse_blocks(write(TABLE))
    assert single[None][2048][0.3] == 28.6


def test_gate_detects_any_changed_cell():
    a = parse_blocks(write(TABLE))[None]
    assert gate(a, a, [0.25])
    b = parse_blocks(write(TABLE.replace("141.6", "141.7")))[None]
    assert not gate(b, a, [0.25])
    c = parse_blocks(write(TABLE.replace("1218.7", "1218.8")))[None]
    assert not gate(c, a, [0.3])  # the off column is always gated


def test_union_drops_off_and_later_rows_win():
    u = union({"off": 9.0, 0.25: 5.0}, {0.25: 4.0, 0.3: 6.0})
    assert u == {0.25: 4.0, 0.3: 6.0}


def test_best_flags_only_pure_range_end_minima():
    assert best({0.15: 3.0, 0.2: 4.0, 0.5: 9.0}, 0.15, 0.5) == (3.0, [0.15], True)
    assert best({0.15: 3.0, 0.175: 3.0, 0.5: 9.0}, 0.15, 0.5) == (3.0, [0.15, 0.175], False)
    assert best({0.15: 5.0, 0.3: 2.0, 0.5: 9.0}, 0.15, 0.5) == (2.0, [0.3], False)
    assert best({0.15: 5.0, 0.3: 4.0, 0.5: 1.0}, 0.15, 0.5)[2]


def test_verdict_boundaries():
    assert "at least the 1.08 margin" in verdict(1.08, 8)
    assert "less than the 1.08 margin" in verdict(1.0799, 8)
    assert "less than the 1.08 margin" in verdict(1.0, 4)
    assert "softmax is ahead at 8x" in verdict(0.9999, 8)



def test_two_table_parser_and_bpb_verdict():
    from score_len_bpb_alpha_grid import parse_two_tables, verdict as vb
    bpb = ("deep-tail bits per byte (same text under every tokenizer):\n"
           "      L        off     a=0.25      a=0.3\n"
           "   1024     1.3806     1.3806     1.3806\n"
           "   2048     2.1572     1.3998     1.4001\n"
           "   4096     2.6963     2.0311     1.6429\n"
           "   8192     3.0344     2.2795     2.1581\n")
    t = parse_two_tables(write("=== a.pt\ntokenizer: gpt2\n" + TABLE.replace("DONE\n", "") + bpb + "DONE\n"))
    assert t["a.pt"]["ppl"][8192][0.3] == 230.0
    assert t["a.pt"]["bpb"][8192][0.3] == 2.1581
    assert t["a.pt"]["bpb"][2048]["off"] == 2.1572
    assert "at least the 1.05 margin" in vb(1.05, 8)
    assert "less than the 1.05 margin" in vb(1.0499, 8)
    assert "softmax is ahead" in vb(0.99, 4)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
    print("ALL OK")

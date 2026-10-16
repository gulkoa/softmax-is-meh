"""Read-ahead loader (triton/dev_scripts/prefetch_loader.py): the batch
stream must be byte-identical to the synchronous path, and producer
failures must surface in the training thread. CPU only; synthetic shards.

    python scripts/test_prefetch_loader.py
"""
import contextlib
import io
import os
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "triton", "dev_scripts"))
import prefetch_loader as PL  # noqa: E402
import train_gpt2_stieltjes as T  # noqa: E402  (real Shards; imports torch)


def ref_batch(shards, bs, ctx, rng):
    """Shards.batch as of tok32k 37d6617, frozen, minus the device copy."""
    xs = np.empty((bs, ctx + 1), dtype=np.int64)
    for i in range(bs):
        maps = shards.sources[rng.choice(len(shards.sources), p=shards.weights)]
        m = maps[rng.integers(len(maps))]
        off = rng.integers(0, len(m) - ctx - 1)
        xs[i] = m[off:off + ctx + 1]
    return xs


def make_sources(root):
    """Three sources of 3 shards each (last = val), unequal lengths, tokens
    over the full uint16 range."""
    g = np.random.default_rng(0)
    dirs = []
    for name, n_shards in (("web", 3), ("math", 3), ("code", 4)):
        d = os.path.join(root, name)
        os.makedirs(d)
        for i in range(n_shards):
            n = int(g.integers(3000, 20000))
            g.integers(0, 65536, n, dtype=np.uint16).tofile(
                os.path.join(d, f"shard_{i:04d}.bin"))
        dirs.append(d)
    return dirs


def check_identical(shards, bs, ctx, acc, n_steps, seed, threads=8):
    pf = PL.Prefetcher(shards, bs, ctx, acc, n_steps,
                       np.random.default_rng(seed), threads=threads)
    r_ref, r_real = np.random.default_rng(seed), np.random.default_rng(seed)
    for i in range(n_steps * acc):
        got = pf.get()
        assert got.dtype == np.int64 and got.shape == (bs, ctx + 1)
        assert got.flags.c_contiguous and got.flags.writeable
        want = ref_batch(shards, bs, ctx, r_ref)
        x, y = shards.batch(bs, ctx, r_real, "cpu")       # the trainer's own path
        assert np.array_equal(got, want), f"micro-batch {i} differs from frozen batch()"
        assert np.array_equal(got[:, :-1], x.numpy()) and np.array_equal(got[:, 1:], y.numpy()), \
            f"micro-batch {i} differs from Shards.batch"
    try:
        pf.get()
        raise AssertionError("get() after the last step should raise")
    except RuntimeError as e:
        assert "exhausted" in str(e), e
    pf.close()


def main():
    with tempfile.TemporaryDirectory() as root:
        web, math_, code = make_sources(root)
        mix = T.Shards("train", f"web={web}:0.6,math={math_}:0.3,code={code}:0.1")
        assert len(mix.sources) == 3 and abs(mix.weights.sum() - 1) < 1e-12
        # from step 0 and from a resume (fresh rng at seed*100003 + start_step)
        check_identical(mix, bs=4, ctx=64, acc=8, n_steps=25, seed=7 * 100003)
        check_identical(mix, bs=4, ctx=64, acc=8, n_steps=10, seed=7 * 100003 + 1234)
        check_identical(mix, bs=8, ctx=1024, acc=2, n_steps=4, seed=3, threads=32)
        single = T.Shards("train", web)                    # no mix: weights [1.0]
        check_identical(single, bs=4, ctx=64, acc=4, n_steps=10, seed=11)

        # a producer error re-raises in get(), and keeps re-raising
        class Broken(PL.Prefetcher):
            def _read(self, row):
                raise OSError(5, "injected EIO")
        pf = Broken(mix, 4, 64, 8, 5, np.random.default_rng(0), threads=2)
        for _ in range(2):
            try:
                pf.get()
                raise AssertionError("injected error was swallowed")
            except OSError as e:
                assert "injected EIO" in str(e), e
        pf.close()

        # a stalled read becomes a TimeoutError, not a silent hang
        release = threading.Event()

        class Stalled(PL.Prefetcher):
            def _read(self, row):
                release.wait()
                return super()._read(row)
        pf = Stalled(mix, 1, 64, 1, 3, np.random.default_rng(0), threads=1,
                     timeout=0.6, poll=0.2)
        t0 = time.time()
        try:
            pf.get()
            raise AssertionError("stalled producer did not time out")
        except TimeoutError:
            assert time.time() - t0 < 5
        release.set()
        assert pf.get().shape == (1, 65)                   # recovers once released
        pf.close()

        # the default (trainer) mode waits a stall out and logs it
        release.clear()
        pf = Stalled(mix, 1, 64, 1, 3, np.random.default_rng(0), threads=1,
                     warn_every=0.3, poll=0.1)
        assert pf.timeout is None
        buf = io.StringIO()
        threading.Timer(1.2, release.set).start()
        with contextlib.redirect_stdout(buf):
            got = pf.get()
        assert got.shape == (1, 65)
        assert "loader: waited" in buf.getvalue(), buf.getvalue()
        pf.close()
    print("OK")


if __name__ == "__main__":
    main()

"""Read-ahead training batches for train_gpt2_stieltjes.py (stilt.1.2 on).

Shards.batch reads each micro-batch's windows one at a time from memory-
mapped shards on GPFS, in the training thread. On 2026-10-16 a 355M chunk
on a node shared with a heavy neighbour job ran at half speed with the
trainer waiting on GPFS. Cold reads of one 355M step's 512 windows from the
stilt.1.2 mix, timed on the login node, took 2.5-7.0 s serially and
1.0-1.2 s with 32 reads in flight. This loader:
  * draws the windows from the trainer's rng in exactly the order
    Shards.batch does (source, shard, offset for each row; rows, then
    micro-batches), so the batch stream is byte-identical to the
    synchronous path (scripts/test_prefetch_loader.py);
  * reads a whole step's windows with os.pread on a thread pool (pread
    releases the GIL; a page fault inside a memmap copy may hold it and
    stall the training thread's kernel launches);
  * runs up to `depth` steps ahead of training in a daemon thread that
    makes no CUDA calls; a producer error is re-raised by get().
Validation keeps the synchronous path (its own rng).

The producer draws ahead of training, so an rng state saved with a
checkpoint would be ahead of the step. This trainer reseeds on resume
(seed * 100003 + start_step) and saves no rng state, which is what makes
read-ahead safe here; a --replay-rng trainer would have to finish its
replay before the producer starts.
"""
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def draw_rows(shards, bs, ctx, rng):
    """(source, shard, offset) for bs rows, consuming rng exactly as
    Shards.batch does."""
    rows = []
    for _ in range(bs):
        s = int(rng.choice(len(shards.sources), p=shards.weights))
        maps = shards.sources[s]
        k = int(rng.integers(len(maps)))
        off = int(rng.integers(0, len(maps[k]) - ctx - 1))
        rows.append((s, k, off))
    return rows


class _Failed:
    def __init__(self, exc):
        self.exc = exc


_DONE = object()


class Prefetcher:
    """int64 arrays [bs, ctx + 1], one per micro-batch, for n_steps
    optimizer steps of grad_accum micro-batches each, in the order the
    synchronous loop would draw them from `rng` (which this object then
    owns: nothing else may draw from it)."""

    def __init__(self, shards, bs, ctx, grad_accum, n_steps, rng,
                 threads=32, depth=2, timeout=900.0, poll=1.0):
        self.shards, self.bs, self.ctx = shards, bs, ctx
        self.grad_accum, self.n_steps, self.rng = grad_accum, n_steps, rng
        self.threads, self.timeout, self.poll = threads, timeout, poll
        self.nbytes = 2 * (ctx + 1)                      # uint16 tokens
        self.fds = [[os.open(m.filename, os.O_RDONLY) for m in maps]
                    for maps in shards.sources]
        self.q = queue.Queue(maxsize=max(1, depth) * grad_accum)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="prefetch")
        self.thread.start()

    def _read(self, row):
        s, k, off = row
        fd, n, pos = self.fds[s][k], self.nbytes, 2 * off
        buf = os.pread(fd, n, pos)
        while len(buf) < n:                    # short read: keep going
            more = os.pread(fd, n - len(buf), pos + len(buf))
            if not more:
                raise EOFError(f"short read in {self.shards.sources[s][k].filename} "
                               f"at token {off}")
            buf += more
        return buf

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=self.poll)
                return True
            except queue.Full:
                pass
        return False

    def _run(self):
        try:
            with ThreadPoolExecutor(self.threads) as pool:
                for _ in range(self.n_steps):
                    rows = [r for _ in range(self.grad_accum)
                            for r in draw_rows(self.shards, self.bs, self.ctx,
                                               self.rng)]
                    raw = b"".join(pool.map(self._read, rows))
                    step = np.frombuffer(raw, dtype=np.uint16).reshape(
                        self.grad_accum, self.bs, self.ctx + 1).astype(np.int64)
                    for mb in step:
                        if not self._put(mb):
                            return
            self._put(_DONE)
        except BaseException as e:             # surfaces in get()
            self._put(_Failed(e))

    def get(self):
        """The next micro-batch. Re-raises a producer error; raises
        TimeoutError if nothing arrives within `timeout` seconds and
        RuntimeError once all n_steps are consumed or the producer died."""
        waited = 0.0
        while True:
            try:
                item = self.q.get(timeout=self.poll)
                break
            except queue.Empty:
                waited += self.poll
                if not self.thread.is_alive() and self.q.empty():
                    raise RuntimeError("prefetch producer exited without a batch")
                if waited >= self.timeout:
                    raise TimeoutError(f"no batch from the prefetch producer "
                                       f"in {self.timeout:.0f} s")
        if item is _DONE:
            self.q.put(_DONE)
            raise RuntimeError(f"prefetcher exhausted after {self.n_steps} steps")
        if isinstance(item, _Failed):
            self.q.put(item)
            raise item.exc
        return item

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)
        for row in self.fds:
            for fd in row:
                os.close(fd)

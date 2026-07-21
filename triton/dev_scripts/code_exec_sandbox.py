"""Sandboxed execution-reward for code RL: run candidate code + assert
tests in a resource-limited subprocess; reward = fraction of tests passed.

Safety: separate process, CPU/mem/output rlimits, no network (unshare via
env-empty + socket-blocking sitecustomize is overkill here — tasks are
MBPP-class; the hard limits are the real guard), 2s wall timeout.
"""
import resource
import subprocess
import sys
import tempfile
import textwrap

RUNNER = textwrap.dedent("""
    import sys
    code = open(sys.argv[1]).read()
    tests = open(sys.argv[2]).read().splitlines()
    ns = {}
    try:
        exec(compile(code, "cand.py", "exec"), ns)
    except Exception:
        print("PARSE_FAIL"); sys.exit(0)
    passed = 0
    for t in tests:
        if not t.strip():
            continue
        try:
            exec(compile(t, "test.py", "exec"), dict(ns))
            passed += 1
        except Exception:
            pass
    print(f"PASSED {passed}")
""")


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 31, 1 << 31))   # 2 GB
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def run_tests(code, test_list, setup_code=""):
    """Return (frac_passed, parsed_ok). test_list: list of assert strings."""
    with tempfile.TemporaryDirectory() as td:
        cand = f"{td}/cand.py"
        tst = f"{td}/tests.txt"
        open(cand, "w").write((setup_code + "\n" if setup_code else "") + code)
        open(tst, "w").write("\n".join(test_list))
        run = f"{td}/run.py"
        open(run, "w").write(RUNNER)
        try:
            out = subprocess.run(
                [sys.executable, "-I", run, cand, tst],
                capture_output=True, text=True, timeout=4,
                preexec_fn=_limits, cwd=td, env={"PATH": ""},
            ).stdout
        except subprocess.TimeoutExpired:
            return 0.0, True
    if "PARSE_FAIL" in out:
        return 0.0, False
    for line in out.splitlines():
        if line.startswith("PASSED"):
            return int(line.split()[1]) / max(1, len(
                [t for t in test_list if t.strip()])), True
    return 0.0, True


def reward(code, test_list, setup_code="", gen_len=0, max_len=256):
    frac, parsed = run_tests(code, test_list, setup_code)
    r = frac + (0.1 if parsed else 0.0) - 0.05 * (gen_len / max_len)
    return r, frac


if __name__ == "__main__":
    # smoke
    r, f = reward("def add(a, b):\n    return a + b",
                  ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"])
    print("good:", r, f)
    r, f = reward("def add(a, b):\n    return a - b",
                  ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    print("half:", r, f)
    r, f = reward("def add(a, b: return", ["assert add(1,2)==3"])
    print("parsefail:", r, f)
    r, f = reward("while True: pass", ["assert True"])
    print("timeout:", r, f)

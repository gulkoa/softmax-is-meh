"""
Measures how approximation error (vs float64 ground truth) decreases
with iteration count, for both Newton-Raphson and binary search,
across multiple q values.  Outputs CSV + plot.
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
from stieltjes import (
    stieltjes_torch, stieltjes_bsearch_torch,
    stieltjes, stieltjes_bsearch,
    DEVICE,
)

N_COLS = 4096
N_ROWS = 64
MAX_NR_ITER = 64
MAX_BS_ITER = 64
Q_VALS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


def ground_truth(x, q):
    x64 = x.to(torch.float64)
    y = stieltjes_bsearch_torch(x64, q=q, num_iter=100, eps=1e-15)
    return y.to(torch.float32)


def sweep():
    torch.manual_seed(42)
    x = torch.randn(N_ROWS, N_COLS, device=DEVICE, dtype=torch.float32)

    rows = []
    for q in Q_VALS:
        y_exact = ground_truth(x, q)
        exact_sum_err = (y_exact.sum(dim=-1) - 1.0).abs().max().item()
        print(f"q={q:5.1f}  ground truth sum_err={exact_sum_err:.2e}")

        # Newton-Raphson: 1..MAX_NR_ITER
        for n_iter in range(1, MAX_NR_ITER + 1):
            y = stieltjes_torch(x, q=q, num_iter=n_iter)
            err = (y - y_exact).abs().max().item()
            sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
            rows.append(dict(method='NR', q=q, num_iter=n_iter,
                             max_err=err, sum_err=sum_err))

        # Binary search: 1..MAX_BS_ITER
        for n_iter in range(1, MAX_BS_ITER + 1):
            y = stieltjes_bsearch_torch(x, q=q, num_iter=n_iter)
            err = (y - y_exact).abs().max().item()
            sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
            rows.append(dict(method='BS', q=q, num_iter=n_iter,
                             max_err=err, sum_err=sum_err))

        # Triton NR: 1..MAX_NR_ITER
        for n_iter in range(1, MAX_NR_ITER + 1):
            y = stieltjes(x, q=q, num_iter=n_iter)
            err = (y - y_exact).abs().max().item()
            sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
            rows.append(dict(method='Triton NR', q=q, num_iter=n_iter,
                             max_err=err, sum_err=sum_err))

        # Triton BS: 1..MAX_BS_ITER
        for n_iter in range(1, MAX_BS_ITER + 1):
            y = stieltjes_bsearch(x, q=q, num_iter=n_iter)
            err = (y - y_exact).abs().max().item()
            sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
            rows.append(dict(method='Triton BS', q=q, num_iter=n_iter,
                             max_err=err, sum_err=sum_err))

    return pd.DataFrame(rows)


def plot(df):
    n_q = len(Q_VALS)
    fig, axes = plt.subplots(2, n_q, figsize=(4 * n_q, 7), sharey=True, squeeze=False)

    row_labels = [('PyTorch', 'NR', 'BS'), ('Triton', 'Triton NR', 'Triton BS')]

    for row, (label, nr_name, bs_name) in enumerate(row_labels):
        for col, q in enumerate(Q_VALS):
            ax = axes[row][col]
            sub = df[df['q'] == q]

            nr = sub[sub['method'] == nr_name]
            bs = sub[sub['method'] == bs_name]

            ax.semilogy(nr['num_iter'], nr['max_err'], 'o-', label='Newton-Raphson',
                        markersize=3, color='blue')
            ax.semilogy(bs['num_iter'], bs['max_err'], 's-', label='Binary Search',
                        markersize=2, color='red')

            ax.axhline(y=1e-7, color='gray', linestyle=':', alpha=0.5, label='float32 eps')
            ax.set_xlabel('Iterations')
            if row == 0:
                ax.set_title(f'q = {q}')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f'{label}\nMax abs error vs f64')
            ax.legend(fontsize=7)

    plt.suptitle(f'Convergence: NR vs Binary Search (n={N_COLS})', y=1.02)
    plt.tight_layout()

    out_path = 'convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.show()


if __name__ == '__main__':
    print(f"Device: {DEVICE}\n")
    df = sweep()
    csv_path = 'convergence.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    plot(df)

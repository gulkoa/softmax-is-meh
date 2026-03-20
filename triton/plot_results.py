import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys


def load_csv(path):
    df = pd.read_csv(path)
    # triton benchmark CSVs have leading spaces in column names
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_stieltjes(ax, csv_path):
    df = load_csv(csv_path)
    x_col = df.columns[0]
    for col in df.columns[1:]:
        ax.plot(df[x_col], df[col], label=col)
    ax.set_xlabel(x_col)
    ax.set_ylabel('GB/s')
    ax.set_title('Stieltjes Transform')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def parse_attn_filename(path):
    """Extract config from fused-attention-batch4-head32-d64-fwd-causal=True-... .csv"""
    m = re.search(r'd(\d+)-(\w+)-causal=(\w+)', path)
    if m:
        return f"d{m.group(1)} {m.group(2)} causal={m.group(3)}"
    return path


def plot_attention(ax, csv_path):
    df = load_csv(csv_path)
    x_col = df.columns[0]
    for col in df.columns[1:]:
        ax.plot(df[x_col], df[col], label=col, marker='o', markersize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel('TFLOPS')
    ax.set_title(parse_attn_filename(csv_path))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)


def main():
    stieltjes_csv = glob.glob('stieltjes-performance.csv')
    attn_csvs = sorted(glob.glob('fused-attention-*.csv'))

    n_stieltjes = len(stieltjes_csv)
    n_attn = len(attn_csvs)
    n_total = n_stieltjes + n_attn

    if n_total == 0:
        print("No CSV files found. Run bench_stieltjes.py and/or bench_fused_attn.py first.")
        sys.exit(1)

    # layout: stieltjes gets a wide panel on top, attention plots in a grid below
    if n_attn == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_stieltjes(ax, stieltjes_csv[0])
    elif n_stieltjes == 0:
        cols = min(n_attn, 3)
        rows = (n_attn + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for i, csv_path in enumerate(attn_csvs):
            plot_attention(axes[i // cols][i % cols], csv_path)
        for i in range(n_attn, rows * cols):
            axes[i // cols][i % cols].set_visible(False)
    else:
        cols = min(n_attn, 3)
        rows_attn = (n_attn + cols - 1) // cols
        fig = plt.figure(figsize=(5 * cols, 4 * (1 + rows_attn)))
        gs = fig.add_gridspec(1 + rows_attn, cols, hspace=0.4, wspace=0.3)

        # stieltjes spans the full top row
        ax_s = fig.add_subplot(gs[0, :])
        plot_stieltjes(ax_s, stieltjes_csv[0])

        # attention plots below
        for i, csv_path in enumerate(attn_csvs):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(gs[1 + r, c])
            plot_attention(ax, csv_path)

    plt.tight_layout()
    out_path = 'benchmark_results.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == '__main__':
    main()

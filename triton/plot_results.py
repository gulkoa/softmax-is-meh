import glob
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_attention(ax, csv_path):
    df = load_csv(csv_path)
    x_col = df.columns[0]
    for col in df.columns[1:]:
        ax.plot(df[x_col], df[col], label=col, marker='o', markersize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel('TFLOPS')
    ax.set_title('Fused Attention')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)


def main():
    stieltjes_csv = 'stieltjes-performance.csv'
    attn_csv = 'fused-attention-performance.csv'

    have_stieltjes = glob.glob(stieltjes_csv)
    have_attn = glob.glob(attn_csv)

    if not have_stieltjes and not have_attn:
        print("No CSV files found. Run bench_stieltjes.py and/or bench_fused_attn.py first.")
        sys.exit(1)

    if have_stieltjes and have_attn:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_stieltjes(ax1, stieltjes_csv)
        plot_attention(ax2, attn_csv)
    elif have_stieltjes:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_stieltjes(ax, stieltjes_csv)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_attention(ax, attn_csv)

    plt.tight_layout()
    out_path = 'benchmark_results.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == '__main__':
    main()

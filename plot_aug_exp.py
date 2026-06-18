import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re

BASE_DIR = "/oscar/home/ckrenter/Desktop/lee-lab-thesis-ckrenter/results/aug_exp"

SIGMA_MAP = {'003': '0.003', '005': '0.005', '01': '0.01', '02': '0.02', '05': '0.05'}

SIMPLE_AUG_ORDER = [
    'baseline_single',
    'baseline_double',
    'flip_horizontal',
    'flip_vertical',
    'rotate_90_cw',
    'rotate_90_ccw',
    'rotate_180',
]

ARCH_COLORS = {
    'manet': '#2196F3',  # blue
    'unet':  '#FF7043',  # orange-red
}


def make_label(exp_name):
    if exp_name == 'baseline_single':
        return 'Baseline\nSingle'
    elif exp_name == 'baseline_double':
        return 'Baseline\nDouble'
    elif exp_name == 'flip_horizontal':
        return 'Flip\nHorizontal'
    elif exp_name == 'flip_vertical':
        return 'Flip\nVertical'
    elif exp_name == 'rotate_90_cw':
        return 'Rotate\n90° CW'
    elif exp_name == 'rotate_90_ccw':
        return 'Rotate\n90° CCW'
    elif exp_name == 'rotate_180':
        return 'Rotate\n180°'
    elif exp_name.startswith('gauss_'):
        sigma_key = exp_name.replace('gauss_', '')
        return f'Gaussian Noise\nσ={SIGMA_MAP.get(sigma_key, sigma_key)}'
    elif exp_name.startswith('deform_'):
        m = re.match(r'deform_s(\d+)_p(\d+)', exp_name)
        if m:
            return f'Elastic Deform\nσ={m.group(1)}, pts={m.group(2)}'
    elif exp_name.startswith('contrast_'):
        m = re.match(r'contrast_a([\d.]+)_b(-?\d+)', exp_name)
        if m:
            return f'Contrast\nα={m.group(1)}, β={m.group(2)}'
    return exp_name


def load_csv(csv_path, arch, metric_col, seed_pattern):
    df = pd.read_csv(csv_path)

    def parse_exp(model_name):
        cleaned = re.sub(seed_pattern, '', model_name)
        cleaned = re.sub(f'^{arch}_', '', cleaned)
        return cleaned

    df['exp_name'] = df['model'].apply(parse_exp)
    return df


def best_hyperparam(sub_df, metric_col):
    """Select hyperparameter combo with best (mean - std) score."""
    grouped = sub_df.groupby('exp_name')[metric_col]
    means = grouped.mean()
    stds  = grouped.std().fillna(0)
    score = means - stds
    return score.idxmax()


def collect_plot_data(arch, csv_type):
    """Return ordered list of (label, exp_name, values_array) for one arch/csv_type."""
    if csv_type == 'summary':
        csv_path   = f"{BASE_DIR}/{arch}/summary/{arch}_aug_summary.csv"
        metric_col = 'test_dice'
        seed_pat   = r'_s\d+$'
    else:
        csv_path   = f"{BASE_DIR}/{arch}/summary/{arch}_aug_kfold.csv"
        metric_col = 'val_dice'
        seed_pat   = r'_f\d+$'

    df = load_csv(csv_path, arch, metric_col, seed_pat)
    plot_data = []

    for aug in SIMPLE_AUG_ORDER:
        rows = df[df['exp_name'] == aug]
        if len(rows) == 0:
            continue
        plot_data.append((make_label(aug), aug, rows[metric_col].values))

    for prefix in ('gauss', 'deform', 'contrast'):
        sub = df[df['exp_name'].str.startswith(prefix)]
        if len(sub) == 0:
            continue
        best_exp = best_hyperparam(sub, metric_col)
        rows     = sub[sub['exp_name'] == best_exp]
        plot_data.append((make_label(best_exp), best_exp, rows[metric_col].values))

    return plot_data, metric_col


def annotate_boxes(ax, bp, data, y_range, offset=0, positions=None):
    """Add mean +/- std annotations above each box."""
    for i, d in enumerate(data):
        pos = (positions[i] if positions is not None else i + 1) + offset
        mean_v = np.mean(d)
        std_v  = np.std(d)
        q3 = np.percentile(d, 75)
        iqr = np.percentile(d, 75) - np.percentile(d, 25)
        whisker_top = min(q3 + 1.5 * iqr, np.max(d))
        ann_y = whisker_top + y_range * 0.012
        ax.annotate(
            f'μ={mean_v:.4f}\n±{std_v:.4f}',
            xy=(pos, ann_y),
            ha='center', va='bottom',
            fontsize=7.5,
            fontweight='bold',
            color='#222222',
        )


def make_boxplot(ax, positions, data, color, width=0.55):
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        notch=False,
        widths=width,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
        flierprops=dict(marker='o', markersize=5, alpha=0.6, markeredgewidth=0.8),
        manage_ticks=False,
    )
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    return bp


# Signle arch plots

def plot_single_arch(arch, csv_type):
    if csv_type == 'summary':
        title_sfx  = 'Test Set Results'
        metric_lbl = 'Test Dice Score'
        out_name   = f"{arch}_aug_summary_plot.png"
    else:
        title_sfx  = 'K-Fold Cross Validation'
        metric_lbl = 'Validation Dice Score'
        out_name   = f"{arch}_aug_kfold_plot.png"

    plot_data, _ = collect_plot_data(arch, csv_type)
    n = len(plot_data)

    fig_w = max(22, n * 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, 11), dpi=200)

    labels  = [d[0] for d in plot_data]
    data    = [d[2] for d in plot_data]
    positions = list(range(1, n + 1))

    bp = make_boxplot(ax, positions, data, color='#4472C4', width=0.60)

    all_vals = np.concatenate(data)
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    y_range = y_max - y_min

    annotate_boxes(ax, bp, data, y_range, positions=positions)

    if csv_type == 'kfold':
        ax.text(0.01, 0.99, 'k = 5', transform=ax.transAxes,
                fontsize=13, va='top', ha='left', color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.8))

    ax.set_title(
        f'{arch.upper()} Augmentation Experiments: {title_sfx}',
        fontsize=17, fontweight='bold', pad=18,
    )
    ax.set_ylabel(metric_lbl, fontsize=13)
    ax.set_xlabel('Augmentation Strategy', fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(y_min - y_range * 0.04, y_max + y_range * 0.25)

    plt.tight_layout()
    out_dir  = f"{BASE_DIR}/{arch}/summary"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# combined plots

def plot_combined(csv_type):
    if csv_type == 'summary':
        title_sfx  = 'Test Set Results'
        metric_lbl = 'Test Dice Score'
        out_name   = 'combined_summary_plot.png'
    else:
        title_sfx  = 'K-Fold Cross Validation'
        metric_lbl = 'Validation Dice Score'
        out_name   = 'combined_kfold_plot.png'

    data_manet, _ = collect_plot_data('manet', csv_type)
    data_unet,  _ = collect_plot_data('unet',  csv_type)

    def _generic_label(exp_name):
        if exp_name.startswith('gauss_'):
            return 'Gaussian\nNoise'
        if exp_name.startswith('deform_'):
            return 'Elastic\nDeform'
        if exp_name.startswith('contrast_'):
            return 'Contrast'
        return make_label(exp_name)

    # Replace parametric labels with generic ones so both arches share the same key
    data_manet = [(_generic_label(exp), exp, vals) for _, exp, vals in data_manet]
    data_unet  = [(_generic_label(exp), exp, vals) for _, exp, vals in data_unet]

    manet_by_label = {d[0]: d[2] for d in data_manet}
    unet_by_label  = {d[0]: d[2] for d in data_unet}

    all_labels = []
    seen = set()
    for label, _, _ in data_manet:
        if label not in seen:
            all_labels.append(label)
            seen.add(label)
    for label, _, _ in data_unet:
        if label not in seen:
            all_labels.append(label)
            seen.add(label)

    # aligned entries: (label, vals_manet_or_None, vals_unet_or_None)
    aligned = [(lbl, manet_by_label.get(lbl), unet_by_label.get(lbl))
               for lbl in all_labels]

    n = len(aligned)
    width = 0.35
    group_centers = np.arange(1, n + 1)

    fig_w = max(26, n * 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, 11), dpi=200)

    pos_manet = group_centers - width / 2
    pos_unet  = group_centers + width / 2

    labels = [a[0] for a in aligned]

    # Only plot boxes for positions where that arch has data
    pos_manet_plot  = [pos_manet[i] for i, a in enumerate(aligned) if a[1] is not None]
    vals_manet_list = [a[1] for a in aligned if a[1] is not None]

    pos_unet_plot  = [pos_unet[i] for i, a in enumerate(aligned) if a[2] is not None]
    vals_unet_list = [a[2] for a in aligned if a[2] is not None]

    bp_manet = make_boxplot(ax, pos_manet_plot, vals_manet_list, color=ARCH_COLORS['manet'], width=width)
    bp_unet  = make_boxplot(ax, pos_unet_plot,  vals_unet_list,  color=ARCH_COLORS['unet'],  width=width)

    all_vals = np.concatenate(vals_manet_list + vals_unet_list)
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    y_range = y_max - y_min

    annotate_boxes(ax, bp_manet, vals_manet_list, y_range, positions=pos_manet_plot)
    annotate_boxes(ax, bp_unet,  vals_unet_list,  y_range, positions=pos_unet_plot)

    if csv_type == 'kfold':
        ax.text(0.01, 0.99, 'k = 5', transform=ax.transAxes,
                fontsize=13, va='top', ha='left', color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.8))

    ax.set_title(
        f'MANet vs U-Net Augmentation Experiments: {title_sfx}',
        fontsize=17, fontweight='bold', pad=18,
    )
    ax.set_ylabel(metric_lbl, fontsize=13)
    ax.set_xlabel('Augmentation Strategy', fontsize=13)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(y_min - y_range * 0.04, y_max + y_range * 0.28)

    legend_handles = [
        mpatches.Patch(color=ARCH_COLORS['manet'], alpha=0.75, label='MANet'),
        mpatches.Patch(color=ARCH_COLORS['unet'],  alpha=0.75, label='U-Net'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    out_dir  = BASE_DIR
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    for arch in ('manet', 'unet'):
        for csv_type in ('summary', 'kfold'):
            plot_single_arch(arch, csv_type)

    for csv_type in ('summary', 'kfold'):
        plot_combined(csv_type)

    print("Done.")

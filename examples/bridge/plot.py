import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11,
                     'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10})

TRAIN_COLOR, VAL_COLOR = '#2E86AB', '#A23B72'
GRID_COLOR, FACE_COLOR = '#F0F0F0', '#F8F9FA'
LOSS_GAP_COLOR, ACC_GAP_COLOR = '#FF6B6B', '#4ECDC4'
GRAY_COLOR = 'gray'

FONT_SIZES = {'label': 12, 'title': 14, 'legend': 11, 'small': 10}

LINE_WIDTH = 2.5
BOX_WIDTH = 0.6
LAST_EPOCHS = 10


def _save_and_show(save_path):
    """Helper function to save and show figure"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Plot saved to: {save_path}")
    plt.show()


def _setup_axis(ax, xlabel, ylabel, title, legend_loc='best'):
    """Setup common axis properties"""
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.set_title(title, fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
    ax.legend(loc=legend_loc, fontsize=FONT_SIZES['legend'])
    ax.grid(True, alpha=0.3, linestyle='--', color=GRID_COLOR)
    ax.set_facecolor(FACE_COLOR)


def _plot_loss_or_acc(ax, epochs, train_data, val_data, data_type='loss'):
    """Plot training and validation loss or accuracy"""
    ax.plot(epochs, train_data, label=f'Training {data_type.title()}',
            color=TRAIN_COLOR, linewidth=LINE_WIDTH, alpha=0.9)
    ax.plot(epochs, val_data, label=f'Validation {data_type.title()}',
            color=VAL_COLOR, linewidth=LINE_WIDTH, linestyle='--', alpha=0.9)


def _mark_extremum(ax, epochs, data, is_max=True, label_prefix='', fmt='.3f', offset_y=10, suffix=''):
    """Mark minimum or maximum point on plot"""
    idx = np.argmax(data) if is_max else np.argmin(data)
    x, y = epochs[idx], data[idx]
    ax.scatter(x, y, color=VAL_COLOR, s=100, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f'{label_prefix}{y:{fmt}}{suffix}', xy=(x, y), xytext=(10, offset_y),
                textcoords='offset points', bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.8), fontsize=FONT_SIZES['small'])


def _style_boxplot(bp):
    """Apply consistent styling to boxplot"""
    for patch, color in zip(bp['boxes'], [TRAIN_COLOR, VAL_COLOR]):
        patch.set(facecolor=color, alpha=0.7)
    whiskers = set(bp['whiskers'])
    for item in bp['whiskers'] + bp['caps']:
        item.set(color=GRAY_COLOR, linewidth=1.5, linestyle='--' if item in whiskers else '-')
    for median in bp['medians']:
        median.set(color='black', linewidth=2)


def create_training_plots(data, save_path=None):
    """Create training process visualization plots"""
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_plots.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = data['epochs']

    # 1. Training and Validation Loss
    _plot_loss_or_acc(axes[0, 0], epochs, data['train_loss'], data['val_loss'], 'Loss')
    _setup_axis(axes[0, 0], 'Training Epochs', 'Loss Value', 'Training vs Validation Loss', 'upper right')
    _mark_extremum(axes[0, 0], epochs, data['val_loss'], is_max=False, label_prefix='Min Val Loss: ')

    # 2. Training and Validation Accuracy
    _plot_loss_or_acc(axes[0, 1], epochs, data['train_acc'], data['val_acc'], 'Accuracy')
    all_acc = np.concatenate([data['train_acc'], data['val_acc']])
    axes[0, 1].set_ylim([max(0, np.min(all_acc) - 5), min(100, np.max(all_acc) + 5)])
    _mark_extremum(axes[0, 1], epochs, data['val_acc'], is_max=True, label_prefix='Max Val Acc: ',fmt='.1f', offset_y=-15, suffix='%')
    _setup_axis(axes[0, 1], 'Training Epochs', 'Accuracy (%)', 'Training vs Validation Accuracy', 'lower right')

    # 3. Loss Distribution (last N epochs)
    bp = axes[1, 0].boxplot([data['train_loss'][-LAST_EPOCHS:], data['val_loss'][-LAST_EPOCHS:]],
                             labels=['Training Loss', 'Validation Loss'], patch_artist=True,widths=BOX_WIDTH)
    _style_boxplot(bp)
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('Loss Value', fontsize=FONT_SIZES['label'])
    axes[1, 0].set_title(f'Loss Distribution (Last {LAST_EPOCHS} Epochs)', fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
    axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--', color=GRID_COLOR)
    axes[1, 0].set_facecolor(FACE_COLOR)

    # 4. Overfitting Analysis
    ax4_loss = axes[1, 1].twinx()
    for ax_gap, gap, color, label in [(axes[1, 1], data['train_loss'] - data['val_loss'], LOSS_GAP_COLOR, 'Loss Gap'),(ax4_loss, data['val_acc'] - data['train_acc'], ACC_GAP_COLOR, 'Accuracy Gap')]:
        ax_gap.fill_between(epochs, 0, gap, alpha=0.3, color=color, label=label)
        ax_gap.plot(epochs, gap, color=color, linewidth=2, alpha=0.7)

    axes[1, 1].set_xlabel('Training Epochs', fontsize=FONT_SIZES['label'])
    axes[1, 1].set_ylabel('Loss Gap (Train - Val)', fontsize=FONT_SIZES['label'], color=LOSS_GAP_COLOR)
    ax4_loss.set_ylabel('Accuracy Gap (Val - Train) %', fontsize=FONT_SIZES['label'], color=ACC_GAP_COLOR)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', color=GRID_COLOR)
    axes[1, 1].set_facecolor(FACE_COLOR)
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax4_loss.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=FONT_SIZES['small'])

    plt.tight_layout()
    _save_and_show(save_path)
    return fig

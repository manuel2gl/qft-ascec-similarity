"""Dendrogram and threshold-diagnostic plots.

Two PNGs are written per clustering run:

1. The **dendrogram** — the UPGMA tree with a horizontal line at the cut
   height that produced the partition.
2. The **threshold diagnostic** — the sorted merge-height curve, the applied
   cut, and the alternate (Mojena, standard) thresholds for reference, with
   a Pearson similarity-floor legend.

matplotlib is imported lazily inside the function so the rest of the
clustering pipeline does not need a display to run.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from cosmic_ascec.clustering.thresholds import pearson_similarity_pct


def plot_annotated_dendrogram(
    linkage_matrix: np.ndarray,
    optimal_k: int,
    cut_height: float,
    filename: str,
    title_suffix: str = "",
    conf_labels: Optional[Sequence[str]] = None,
    mojena_threshold: Optional[float] = None,
    mojena_k: Optional[int] = None,
    n_eff: Optional[float] = None,
) -> None:
    """
    Save two plot files:
      1. Dendrogram with horizontal cut line -> filename (e.g., dendrogram.png)
      2. Mojena diagnostic -> threshold_diagnostic.png in the same directory

    Verbatim port of cosmic-v01's ``plot_annotated_dendrogram`` (4543-4723).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as _mticker
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import dendrogram

    # Check if all distances are effectively zero
    lm = linkage_matrix.copy()
    if np.all(lm[:, 2] == 0.0):
        lm[:, 2] += 1e-12

    # --- File 1: Dendrogram ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    dendrogram(lm, labels=conf_labels, leaf_rotation=90, leaf_font_size=11, ax=ax1)

    cut_label = rf'Threshold $\tau$={cut_height:.2f} ($n_c$={optimal_k})'
    ax1.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=cut_label)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_title(f"Hierarchical Clustering Dendrogram ({title_suffix})", fontsize=16)
    ax1.set_xlabel("Configuration", fontsize=18)
    ax1.set_ylabel("UPGMA linkage distance", fontsize=18)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.yaxis.set_major_formatter(_mticker.FormatStrFormatter('%.1f'))
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    fig1.savefig(filename, dpi=150)
    plt.close(fig1)

    # --- File 2: Diagnostic plot (sorted merge heights + knee, applied cut, Mojena) ---
    heights_sorted = np.sort(linkage_matrix[:, 2])
    n_merges = len(heights_sorted)
    if n_merges < 3:
        return

    diag_filename = os.path.join(os.path.dirname(filename), "threshold_diagnostic.png")

    # Match dpi to savefig dpi so on-canvas window_extent measurements
    # reflect the size of the rendered image.
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    merge_indices = np.arange(1, n_merges + 1)

    # Insert the exact cut-crossing point so the below-cut (blue) and
    # above-cut (red) fills pivot on the same vertex — no white notch.
    xf = merge_indices.astype(float)
    hf = heights_sorted.astype(float)
    _cross = np.where((heights_sorted[:-1] <= cut_height)
                      & (heights_sorted[1:] > cut_height))[0]
    if len(_cross):
        _i = int(_cross[0])
        _frac = (cut_height - heights_sorted[_i]) / (heights_sorted[_i + 1] - heights_sorted[_i])
        _xc = merge_indices[_i] + _frac * (merge_indices[_i + 1] - merge_indices[_i])
        xf = np.insert(xf, _i + 1, _xc)
        hf = np.insert(hf, _i + 1, cut_height)

    # Below-cut fill (blue) — area under the curve, capped at the cut line.
    ax2.fill_between(xf, 0, np.minimum(hf, cut_height), alpha=0.15,
                     color='#3498db', linewidth=0, edgecolor='none')
    ax2.plot(merge_indices, heights_sorted, 'o-', color='#3498db', linewidth=0.9,
             markersize=3, label='Merge heights')

    # Applied cut (red dashed) — this is the threshold the run actually used.
    n_above_cut = int(np.sum(heights_sorted > cut_height))
    ax2.axhline(y=cut_height, color='#e74c3c', linestyle='--', linewidth=2,
                label=rf'Applied cut $\tau$={cut_height:.2f} ($n_c$={n_above_cut + 1})')

    # Standard τ=2.0 reference — listed in the legend for comparison, not drawn.
    STANDARD_T = 2.0
    n_standard = int(np.sum(heights_sorted > STANDARD_T)) + 1
    ax2.plot([], [], ' ',
             label=rf'Standard $\tau$=2.00 ($n_c$={n_standard})')

    # Mojena reference — listed in the legend for comparison, not drawn on the axes.
    if mojena_threshold is not None:
        mojena_label = rf'Mojena $\tau$={mojena_threshold:.2f}'
        if mojena_k is not None:
            mojena_label += rf' ($n_c$={mojena_k})'
        ax2.plot([], [], ' ', label=mojena_label)

    # Shade between-cluster region (above applied cut), interpolated at the cross.
    ax2.fill_between(xf, cut_height, hf,
                     where=(hf >= cut_height), interpolate=True,
                     alpha=0.2, color='#e74c3c', linewidth=0, edgecolor='none',
                     label='Between-cluster merges')

    ax2.set_xlabel("Merge Step (sorted)", fontsize=15)
    ax2.set_ylabel("UPGMA linkage distance", fontsize=15)
    ax2.set_title("Threshold Diagnostic (Merge Height Distribution)", fontsize=16)
    ax2.tick_params(labelsize=13)
    ax2.yaxis.set_major_formatter(_mticker.FormatStrFormatter('%.1f'))
    leg_main = ax2.legend(loc='upper left', fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # Pearson similarity (for the applied threshold and a reference threshold).
    trust_segments = []
    if n_eff is not None and n_eff > 0:
        applied_is_standard = abs(cut_height - STANDARD_T) <= 1e-6

        def _fmt_trust_segment(label, t_val):
            pct = pearson_similarity_pct(t_val, n_eff)
            if pct is None:
                return f"{label} τ={t_val:.2f} → N/A"
            return f"{label} τ={t_val:.2f} → {pct:.1f}%"

        # Always show applied; show standard only when it differs; always show Mojena if available.
        trust_segments.append(_fmt_trust_segment("Applied", cut_height))
        if not applied_is_standard:
            trust_segments.append(_fmt_trust_segment("Standard", STANDARD_T))
        if mojena_threshold is not None:
            trust_segments.append(_fmt_trust_segment("Mojena", float(mojena_threshold)))

    # Run tight_layout BEFORE measuring the main legend.
    fig2.tight_layout()

    if trust_segments:
        # Anchor the second legend just to the right of the main legend.
        fig2.canvas.draw()
        ax2.add_artist(leg_main)
        leg_bbox_axes = leg_main.get_window_extent().transformed(
            ax2.transAxes.inverted())
        trust_anchor_x = min(leg_bbox_axes.x1 + 0.005, 0.55)
        trust_anchor_y = leg_bbox_axes.y1

        _trust_kwargs = dict(
            loc='upper left',
            bbox_to_anchor=(trust_anchor_x, trust_anchor_y),
            bbox_transform=ax2.transAxes,
            fontsize=12,
            title='Similarity floor',
            title_fontsize=12,
            handlelength=0, handletextpad=0,
            borderaxespad=0,
            borderpad=0.4,
        )
        _trust_handles = [Patch(visible=False) for _ in trust_segments]
        trust_leg = ax2.legend(handles=_trust_handles, labels=trust_segments,
                               **_trust_kwargs)

        # Iterate the trust legend's borderpad until both boxes reach the
        # same pixel height.
        fontsize_px = 12 * fig2.dpi / 72
        fig2.canvas.draw()
        trust_bp = 0.4
        for _ in range(12):
            h_main = leg_main.get_window_extent().height
            h_trust = trust_leg.get_window_extent().height
            delta_px = h_main - h_trust
            if abs(delta_px) <= 0.5:
                break
            new_bp = max(0.1, trust_bp + delta_px / (2 * fontsize_px))
            if abs(new_bp - trust_bp) < 0.002:
                break
            trust_leg.remove()
            _trust_kwargs['borderpad'] = new_bp
            trust_leg = ax2.legend(
                handles=[Patch(visible=False) for _ in trust_segments],
                labels=trust_segments, **_trust_kwargs)
            trust_bp = new_bp
            fig2.canvas.draw()

    fig2.savefig(diag_filename, dpi=150)
    plt.close(fig2)


__all__ = ["plot_annotated_dendrogram"]

"""Test helper: level-0 FC with the window given in seconds (the retired
single-level ``compute_fc`` surface).

Delegates to ``compute_fc_levels``: no decimation; prewhitening follows the
library's shipped default unless passed. The aurora package keeps the same
helper in ``emptyui-aurora/tests/_synthetic.py``.
"""

from __future__ import annotations


def compute_fc_seconds(value, input_channels, output_channels, window_length_s,
                       overlap, window_type, **kw):
    from emptyui_aurora._processing import compute_fc_levels, to_channels

    _, sample_rate, _, _ = to_channels(value, check_finite=False)
    levels = compute_fc_levels(
        value, input_channels, output_channels, num_decimations=0,
        num_samples_window=max(8, int(window_length_s * sample_rate)),
        overlap=overlap, window_type=window_type, **kw)
    if 0 not in levels:
        raise ValueError(
            f"series shorter than one window ({window_length_s} s at "
            f"{sample_rate} Hz): compute_fc_levels returned no level 0")
    return levels[0]

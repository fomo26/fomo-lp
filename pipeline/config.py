# Fairness-variable configuration — fill in the three values below.
from __future__ import annotations


# PLEASE SPECIFY: first demographic variable column in your subjects CSV.
FEATURE1_CSV_COL = ""

# PLEASE SPECIFY: second demographic variable column in your subjects CSV.
FEATURE2_CSV_COL = ""

# PLEASE SPECIFY: bin upper bounds (inclusive) and integer group labels for FEATURE1.
# Format: tuple of (upper_bound, bin_label) pairs in ascending order.
# The last upper_bound must be large enough to cover all expected values.
# Example: FEATURE1_BINS = ((25, 0), (50, 1), (75, 2), (1000, 3))
FEATURE1_BINS: tuple[tuple[int, int], ...] = ()

"""Data governance: upload, masking, quality scoring, and 7:1.5:1.5 splitting.

Plan: M1. The split logic here is the only place the "synthetic rows never enter
the test split" invariant is enforced; it filters on `son_contracts.DataOrigin`.
"""

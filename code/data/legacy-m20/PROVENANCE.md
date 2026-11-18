# Legacy M=20 scaling inputs

Recovered 19 November 2026 from `simulation/gravity/q5a-o5/o25_outputs/q{29,61,101,151,211}_o25.npz`.
These are compact field-for-field exports, not new simulations. The original files total about 14 GB;
unrelated stored block arrays are omitted. The exporter copies the exact numerical arrays used by
`o25_scaling_plot.py`, including q, M_per_pair, n0, n1, delta_pair_global, r2_global,
delta_pair_mean, delta_pair_std, sigma_pair_mean and ns. See `export_legacy_m20.py`.
The original generator on these fields reproduces the committed legacy PDF's extracted text and every
pixel at 100 dpi (0 differing pixels). Its global extrapolated intercept is 5.655, displayed as 5.66.
This M=20 dataset is distinct from the M=50 production campaign under `data/extended/`.
SHA256SUMS records the compact input files; build.sh verifies them before plotting.

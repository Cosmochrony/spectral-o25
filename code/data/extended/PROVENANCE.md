# Inputs of the reconstructed extended-campaign figure

These files are the inputs read by `code/o25_scaling_extended_reconstruct.py`. They were copied unchanged on
18 November 2026 from the programme workspace `simulation/spectral/o25/`; `SHA256SUMS` lists their digests,
which the script verifies before reading them.

| Files | Origin | Status at copy time |
|---|---|---|
| `q{29,61,101,151,211}_o25.npz` | `simulation/spectral/o25/o25_outputs/` | tracked in the archived `simulation` repository (Cosmochrony/simulation, HEAD `c383f42`; last data commits `b1b3765`, `ce0212f`, April 2026), identical to the committed blobs |
| `dpc_q{307,401,601}.npz` | `simulation/spectral/o25/dpc_out/` | untracked local files, written 1-3 July 2026 by `delta_pair_convergence_campaign.py` |
| `n1_q{307,401,601}.json` | `simulation/spectral/o25/n1_scaling_out/` | untracked local files, written 29-30 June 2026 (canonical single-block windows) |

All inputs predate the published figure (created 3 July 2026, 22:39 CEST).
The capped-BFS intermediates `bfs_q{307,401,601}.npz` (about 85 MB in total) are not needed for the figure or
the tabulated values; they remain in the workspace `simulation/spectral/o25/dpc_out/` and are not distributed.

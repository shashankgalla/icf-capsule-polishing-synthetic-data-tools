# Release validation

| Check | Status | Evidence or remaining requirement |
| --- | --- | --- |
| Synthetic-only export | PASS | Allowlisted frames, CSV, sanitized metadata; no weights or real data |
| Exported image integrity | PASS | 16,000 decoded RGB frames, 480x48 |
| Historical independent trajectories | FAIL | Simulation and seed sharing across splits |
| Historical cross-split exact frames | PASS | No interval or exact content overlap |
| New simulation-disjoint split | PASS | Strict identity, seed, interval and exact content audit |
| Clean CPU installation | PASS | Isolated Python3.13 venv, pinned packages, pip check |
| Automated tests | PASS | 12 tests; see saved final test log |
| Actual clean CPU training | PASS | One active-physics optimizer step, checkpoint save |
| GPU training/inference/resume | PASS, bounded | Exact adapter/encoder resume and repeated inference equality; rank2, 4 frames |
| Clean CPU inference | PASS | Saved four-frame NPY and MP4 |
| Evaluation CLI full coverage | PASS, fixture | All 54 validation records, zero-array fixture; not model scores |
| Fixed crop CLI | PASS | Independent numeric synthetic image fixture |
| Complete test PSNR/SSIM benchmark | NOT VERIFIED | Only bounded paired smoke comparison executed |
| Full paper benchmark reproduction | NOT VERIFIED | No independent full training or motion/FID/FVD suite |
| Fresh Blender rendering | NOT VERIFIED | Blender unavailable |
| Fixed crop equivalence | PASS, limited | First frame of each simulation only |
| Fresh CUDA installation | NOT VERIFIED | GPU checks used existing CUDA environment |
| Full-size training / multiple GPUs | NOT VERIFIED | Small single-process checks only |
| Code/data licenses | PASS | Owner confirmed rights 2026-11-06; code released under MIT (see `LICENSE`) |
| Third-party license compatibility | NOT VERIFIED | Formal review outstanding |
| Public dataset/checkpoint URLs | PASS | Dataset public at huggingface.co/datasets/shashankgalla/synthetic-capsule-polishing; no checkpoints in this package |
| Local upload package | PASS | Folder dataset, card, schema, checksums and loader |
| Public publication readiness | PARTIAL | Licenses resolved 2026-11-06; historical split leakage and full-scale benchmark validation remain open (see rows above) |

Nothing has been uploaded or pushed. Local packaging does not imply approval to publish.

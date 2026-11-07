# Synthetic reproducibility audit

Scope: synthetic generation, loading, splits, training, checkpointing, inference and evaluation. Original source and data were preserved. No real data or detector weights were used. Findings refer to the supplied snapshot, not every historical experiment.

| Severity | Evidence in supplied source | Implemented response | Scientific implication |
| --- | --- | --- | --- |
| Critical | `src/dataset.py` dropped slip; samples defaulted to .95 | Strict label retention including zero and finite-range checks | Correct conditioning changes historical training inputs |
| Critical | Missing split CSV fell back to complete metadata | Missing split now raises before model loading | Prevents accidental split contamination |
| High | Historical splits share all simulation identities and seeds | Preserved historical manifests; added explicit whole-simulation protocol and preflight audit | Historical independence claim fails; new split needs new results |
| High | Physics loss used theta60, legacy radii, dimensionally mixed effective speed and FFT ratio proxy | Separate explicit manuscript profile with physical units, theta45 and stratified ratio MSE | Cannot silently equate supplied implementation with paper method |
| High | Learnable frequencies and cross-attention Q adapters differed from manuscript | Explicit manuscript profile freezes frequency scale and excludes cross-attention Q adapters | Changes trainable architecture and results |
| High | Nominal remapped RPM used for physical conditioning | Manuscript profile uses raw RPM21..108 | Changes inputs and physical losses |
| High | Missing checkpoint components could fall back to random/default modules | Inference requires weights and saved architecture settings | Prevents misleading inference runs |
| High | Checkpoints lacked optimizer, random states and scaler | Full epoch state and strict configuration checks; single-process constraint | Exact resume is tested only in the bounded configuration |
| High | “FVD” used r3d_18 and silent optical-flow fallback | Excluded unverified metric, explicit PSNR/SSIM evaluator only | Full reported video metrics remain unverified |
| Medium | Crop fitting used complete-trajectory ground truth | Explicit fixed-crop preprocessor; exported pixels retained | Future generation must declare crop without held-out fitting |
| Medium | Stale absolute paths and inconsistent GT NPZ scale | Relative paths, metadata allowlist, scaled CSV only | No original data overwritten |
| Medium | Best checkpoint uses training loss; source eval can sweep test checkpoints | Documented criterion and separate val/test evaluator | No evidence establishes historical tuning independence |

## Executed evidence

All 16,000 exported images decoded with expected dimensions. Exact frame hashing found no cross-split duplicates. Historical temporal protocol fails strict simulation/seed separation. The new protocol passes frame, seed and simulation checks. Exact hashing does not detect every near duplicate.

An isolated Python3.13 CPU virtual environment installed the pinned dependencies, passed `pip check`, imported train/inference modules and passed 12 automated tests. Actual SVD training ran one optimizer step on CPU with active physics losses and saved checkpoints. GPU execution used the existing PyTorch2.10 CUDA12.8 environment on GPU0, one RTX4090. Bounded runs used rank2, four frames, 64x32 pixels, batch2 and two clips. These are execution diagnostics, not paper experiments. The final GPU integration passed exact adapter/encoder tensor equality after resume and repeated inference array equality. Clean CPU inference also passed. The evaluation CLI processed all 54 validation records using explicitly artificial zero arrays; this verifies execution and coverage only. Inspect accompanying evidence JSON and logs for commands.

An initial inference regression in the candidate was fixed. Initial resume comparison failed; saving the AMP scaler and enabling deterministic kernels corrected exact adapter and encoder tensor equality in the subsequent run. The validation driver also had a package import failure after inference; that was fixed and rerun. Failed evidence is retained rather than suppressed.

Preprocessing equivalence was checked for one frame from each of 20 simulations, not every raw image. Blender rendering was not run. Full-size rank32 training, complete epochs on all clips, the complete paper metrics and reported benchmark comparisons were not executed. No smoke metric is evidence that the accepted results reproduce.

## Paper correspondence and unresolved questions

The manuscript profile implements the declared Fourier, adapter and physics choices separately from the supplied legacy implementation. SVD/VAE pixel conventions remain inherited and require further scientific confirmation. The training slip grid differs from the evaluation sweep described in the paper. Detector provenance, exact I3D feature extractor and evaluation preprocessing must be established before reproducing motion/FID/FVD results. A clean CUDA installation, distributed execution and fresh simulator rendering remain unverified.

The historical dataset has one trajectory per condition pair and cannot provide independent replicate seeds for the same held-out conditions. New independent simulations would strengthen evaluation but would change the dataset and are not fabricated here. Code/data licensing and third-party compatibility require owner confirmation. Public repository IDs and model release approvals were not supplied.

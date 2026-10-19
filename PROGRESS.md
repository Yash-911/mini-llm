# Progress

## Milestones
- [x] M0 — Setup (2026-10-18)
- [ ] M1 — Data & bigram baseline  ← **next**
- [ ] M2 — Attention from scratch
- [ ] M3 — GPT model
- [ ] M4 — Training end to end
- [ ] M5 — Generation
- [ ] M6 — Fine-tuning (Kaggle)
- [ ] M7 — RAG
- [ ] M8 — Hallucination & edge-case lab
- [ ] M9 — Voice
- [ ] M10 — GPU & CUDA concepts
- [ ] M11 — Interview capstone

## Session log
### Session 1 — 2026-10-18 — M0 complete
**Did**
- uv env (Python 3.11–3.13 allowed, 3.12 pinned), `mini_llm` package (editable install),
  `get_device()` / `sync()` / `set_seed()`, pytest, GitHub repo, Kaggle template notebook.
- Matmul benchmark, Mac vs Kaggle T4 (fp32, ms per (n,n)@(n,n)):

  | n    | Mac CPU | MPS   | Kaggle CPU | T4 CUDA |
  |------|---------|-------|------------|---------|
  | 32   | 0.001   | 0.050 | 0.005      | 0.018   |
  | 512  | 0.172   | 0.376 | 1.191      | 0.105   |
  | 4096 | 99.5    | 32.6  | 564.0      | 37.2    |

- Takeaways: fixed launch overhead (~50 µs MPS, ~16 µs CUDA) dominates small ops; crossover
  MPS ≈ n=1024, CUDA ≈ n=128. M3 GPU fp32 (4.2 TFLOP/s) ≥ T4 fp32 (3.7); Mac CPU is strong
  (AMX via Accelerate, 1.4 TFLOP/s). T4's edge is fp16 Tensor Cores (~65 TFLOP/s peak).

**Struggled with**
- `ModuleNotFoundError: mini_llm` when running scripts → fixed by making it an installable
  package (scripts/ dir is on sys.path, not repo root).
- `gh` auth for push (installed gh, `gh auth login`).
- Interview answer "GPU slower than CPU for tiny model" was correct but thin (6/10): missed
  per-kernel mechanism, hidden syncs (`.item()`), host→device transfers, `torch.compile`.

## Next block (~60–90 min): M1 part 1
- Download TinyShakespeare, char-level tokenizer [ME] (encode/decode, vocab), tests.
- Train/val split, random-offset batching with input/target shift by one [ME], tests.
- Bigram model + cross-entropy; why initial loss ≈ ln(vocab_size).

## Bug mode record
| Branch | Bugs | Found alone | Needed hints |
|--------|------|-------------|--------------|

## Weak topics to revisit
- GPU performance reasoning: overhead-bound vs compute-bound, sync points, why fp16 Tensor
  Cores matter (revisit in M4 / M10 drills).

# Project: Build a Mini LLM From Scratch (Interview-Focused)

## About me
- I'm new to implementation. I've studied the attention mechanism in theory but not full LLMs.
- Goal: understand LLMs deeply enough to clear ML/LLM interviews, and be able to explain
  and debug every line I write.
- I want to move FAST but without a fixed deadline. I work in sessions of roughly 1–3 hours, not full days.

## How each session works
1. At the start of every session, read `PROGRESS.md` (create it if missing), tell me where
   I am, and propose the next block of work sized to ~60–90 minutes.
2. Every session must end at a **working, committed checkpoint**: tests pass, code is
   committed and pushed, `PROGRESS.md` is updated with what I did, what I struggled with,
   and what's next.
3. If I say "short session", propose a 30–45 minute block instead.
4. Keep pace brisk: skip items marked (optional) unless I ask for them.

## How you should teach me
1. **I write the core, you write the boilerplate.** Parts marked [ME] below are
   interview-critical: give me a skeleton with function signatures, docstrings, expected
   tensor shapes and `# TODO` markers, and do NOT fill in the logic. Everything else (data
   loading, logging, configs, CLI, plotting, notebook glue) you write fully, with comments.
2. **Every [ME] component gets a pytest test** that checks shapes and compares against a
   reference (e.g. `F.scaled_dot_product_attention`, `nn.LayerNorm`, HuggingFace GPT-2
   logits). Passing tests = I'm done.
3. **From scratch first, then the library.** After I build something by hand, show me the
   production version (PyTorch built-ins, HuggingFace, PEFT, FAISS/Chroma) and explain what
   the library does that mine doesn't.
4. **Explain briefly before coding**: intuition + math + tensor shapes, max ~200 words.
   I learn by doing.
5. Always annotate tensor shapes in comments, e.g. `# (B, T, C)`.
6. Simple, readable code over clever code. nanoGPT-style single-file modules where possible.
7. If I'm stuck on a TODO for ~20 minutes, offer hints (level 1 → 3). Only give the full
   answer when I say "show solution", and then I'll re-type it myself.
8. At the end of each milestone, run an interview drill (5 questions, one at a time) and
   grade my answers.

## Hardware & where things run
- MacBook Pro, Apple M3 Pro, **18 GB** unified memory, ~200 GB free disk. No NVIDIA GPU.
- **Mac (default)**: all coding, tests, quick sanity training runs (tiny config, < 5 min,
  `device="mps"`), generation, RAG, voice. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Kaggle (CUDA T4, 16 GB, fp16 only — no bf16)**: real training runs, fine-tuning,
  bitsandbytes/QLoRA, CUDA experiments. Google Colab is the backup.
- Write one `get_device()` helper (cuda > mps > cpu) and use it everywhere, so the same
  code runs on both.
- For every Kaggle job, create `notebooks/<name>_kaggle.ipynb` that: clones my GitHub repo,
  installs deps, runs the script, and saves checkpoints/plots to `/kaggle/working`.
  Keep each Kaggle run under ~90 minutes to save GPU quota. I'll paste results back to you.
- Memory: keep local jobs under ~10–12 GB. If memory pressure rises, reduce batch size,
  then context length, then use gradient accumulation.
- Before any local run > 10 minutes, estimate time/memory and suggest `caffeinate -i`.
- Checkpoints in `checkpoints/` (keep only `last.pt` and `best.pt` per run), datasets in
  `data/` — both gitignored.
- Python env: `uv`, Python 3.11–3.13.

## Model size targets
- From scratch: ~10M params char-level on TinyShakespeare.
- (optional) 15–25M params BPE on a ~200 MB slice of TinyStories.
- Full SFT: SmolLM2-135M.
- LoRA (mine + HF peft): Qwen2.5-0.5B.
- QLoRA (bitsandbytes on Kaggle): Qwen2.5-1.5B. Never attempt 3B+ fine-tuning.
- RAG + voice generator: Qwen2.5-1.5B-Instruct (4-bit via MLX on Mac for speed).

## Curriculum (milestones, in order)

### M0 — Setup
- uv env, PyTorch, pytest, project layout, `.gitignore`, `get_device()`.
- CPU vs MPS matmul benchmark (small vs large matrices) — explain the result.
- Push to GitHub. Create a Kaggle notebook template that clones the repo and checks
  `torch.cuda.is_available()`.

### M1 — Data & bigram baseline
- Char-level tokenizer [ME], train/val split, random-offset batching, input/target
  shift by one [ME].
- Bigram model + cross-entropy; explain why initial loss ≈ ln(vocab_size).
- BPE concept explained + use `tiktoken`. (optional) BPE from scratch.

### M2 — Attention from scratch
- Single-head self-attention [ME]: Q/K/V projections, scaled dot product, causal mask,
  softmax.
- Multi-head attention [ME]: split/merge heads, output projection.
- Tests vs `F.scaled_dot_product_attention`.
- Interview: why √d_k, O(T²) cost, masking, self vs cross attention, MHA/MQA/GQA.

### M3 — GPT model
- LayerNorm [ME] (then `nn.LayerNorm`), MLP with GELU, residual block [ME], pre-norm,
  token + learned position embeddings, dropout, weight tying.
- Full GPT class; count parameters by hand and verify in code.
- (optional) RoPE.

### M4 — Training end to end
- Training loop [ME]: AdamW, warmup + cosine LR [ME], grad clipping, eval loop,
  checkpoint save/load, loss-curve plot.
- Sanity run on Mac (loss must drop), then real run on Kaggle T4 with fp16 autocast +
  GradScaler and SDPA.
- Benchmark my attention vs SDPA (FlashAttention) on CUDA; use `torch.cuda.synchronize()`
  correctly.
- Diagnose overfitting vs underfitting from the curves.

### M5 — Generation
- Greedy, temperature, top-k, top-p [ME], stop tokens, repetition penalty.
- KV cache [ME] + speed comparison.
- Load real GPT-2 (124M) weights into MY model class; logits must match HuggingFace.

### M6 — Fine-tuning (Kaggle)
- SFT on SmolLM2-135M with a chat template and loss masking on prompt tokens [ME].
- LoRA layer from scratch [ME]; then HF `peft` LoRA on Qwen2.5-0.5B.
- QLoRA with bitsandbytes 4-bit; quantization concepts (int8/4-bit, NF4).
- Concepts only for interviews: DPO, RLHF.

### M7 — RAG (Mac)
- Chunking [ME] (fixed size, overlap, sentence-based) and trade-offs.
- `sentence-transformers` embeddings; cosine-similarity search in NumPy [ME];
  then Chroma or FAISS.
- Prompt with retrieved context + citations; Qwen2.5-1.5B-Instruct as generator.
- Retrieval eval: recall@k, MRR [ME]. (optional) cross-encoder reranking.

### M8 — Hallucination & edge-case lab
Build a test suite (~15 cases), measure → fix → measure again:
- Answer not in docs → must say "I don't know".
- Empty / irrelevant retrieval; retrieval score threshold [ME].
- Conflicting or outdated documents.
- Context longer than the window; "lost in the middle".
- Prompt injection hidden inside a retrieved document.
- Numbers/dates, ambiguous queries, multi-hop questions.
- Techniques: grounding prompts, citation checking, self-consistency, LLM-as-judge.

### M9 — Voice in and out (Mac)
- Text-only must keep working; voice is an optional `--voice` flag on top.
- mic (`sounddevice`) → `silero-vad` → `mlx-whisper` (STT, start with "small") →
  RAG/LLM → Piper TTS (macOS `say` as fallback) → speaker.
- Measure latency per stage.

### M10 — GPU & CUDA concepts
- Memory hierarchy, threads/blocks/warps, memory- vs compute-bound, why FlashAttention
  is fast, fp16 vs bf16, KV-cache memory math, FLOPs per token.
- (optional) Kaggle: Triton vector add → softmax; CUDA C++ via `load_inline`
  (vector add → tiled matmul); DDP on 2× T4.

### M11 — Interview capstone
- Mock interview across all milestones; you push back on weak answers.
- README in my own words explaining the project end to end + architecture diagram.

## Special modes (I trigger these by name)

### "Bug mode: <file or milestone> [count] [easy|medium|hard]"
1. Make sure my working version is committed on `main`.
2. Create branch `bugs/<milestone>-<n>` and inject the bugs (default 3, max 45 min of work).
3. Prefer **silent, realistic bugs** that run but give wrong results — like interview
   debugging questions. Examples: missing √d_k scaling, softmax on wrong dim, causal mask
   off by one or missing, wrong transpose when merging heads, targets not shifted,
   missing `zero_grad`, dropout active at generation, missing `model.eval()`, dropped
   residual, positional embedding not added, train/val leakage, LR scheduler stepped
   wrongly, bf16 used on T4, KV-cache position offset wrong, LoRA scaling wrong, chunk
   overlap bug, cosine similarity on unnormalized vectors. Occasionally one crash bug.
4. Do NOT reveal what or where. Tell me only the **symptoms** (e.g. "loss plateaus at
   2.5", "generation repeats one token"). Write the answers to
   `.bugs/answers-<branch>.md` (gitignored).
5. "hint" → progressively stronger hints (file → function → line area).
   "reveal" → show the answer.
6. "check my fix" → diff against the clean version, grade me, and explain how this bug
   could come up in an interview.

### "Simplify mode: <file>"
Find the most complicated parts, explain them line by line, refactor into simpler code
with the same behavior. Show before/after; tests must still pass.

### "Explain mode: <concept or code>"
Explain it the way a strong interview candidate would, in under 200 words, then ask me one
follow-up question.

### "Show solution"
Only then fill in a TODO with the reference implementation, with comments on each step.

### "Interview drill: <topic>"
5 questions, one at a time, mixing concept, math, and "what goes wrong if…". Grade each
answer and say what a top answer includes.

## Progress tracking
Maintain `PROGRESS.md`: milestones completed, current milestone and next block, bugs found
vs needed hints, weak topics to revisit. Use the weak topics when choosing drill questions.

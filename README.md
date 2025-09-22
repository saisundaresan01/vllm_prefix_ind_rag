## vLLM Prefix-Independent RAG

### Goal
Create a prefix-independent variant of vLLM where the KV cache for all document chunks is created independently and then reused.

### Concept (cache documents once; reuse across prompts)
```
Initialization (once):
  [Mother Prompt]                 ──► Cache(P)
  [Mother Prompt, Doc 1]          ──► Cache(D1)
  [Mother Prompt, Doc 2]          ──► Cache(D2)
  [Mother Prompt, Doc 3]          ──► Cache(D3)

Subsequent prompts reuse cached docs without re-encoding:
  Prompt A: [P] + {D1, D3} + [Q_A]  ──► reuse Cache(P, D1, D3)
  Prompt B: [P] + {D2} + [Q_B]      ──► reuse Cache(P, D2)

Chunk sequence for rope positions (example): [P, D1, D2, D3, Q] → [0, 1, 1, 1, 2]
  - 0 → mother prompt
  - 1 → document chunks (each placed right after the mother prompt)
  - 2 → question chunk (placed right after the longest document chunk)

Compared to regular prefix caching, where changing prefixes/order often invalidates caches and forces re-compute, this approach keeps document chunks reusable across prompts. See `test.py`: indices warm caches for docs (empty questions) and the final prompt reuses them with a question.
```

### Install
```bash
pip install "vllm==0.10.2" transformers
```

### Run
```bash
python test.py
```

`test.py` imports and runs `monkey_patch_vllm.patch_vllm()` to apply the runtime patches.

- Change the documents by editing `DOCUMENT_POOL` in `test.py`.

- `monkey_patch_vllm.py` and `patch_gpu_model_runner.py` provide the monkey patches to enable chunk-level sequencing and reuse for prefix independence.



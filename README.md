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

Subsequent prompts reuse cached docs KV cache:
  Prompt A: [P] + {D1, D3} + [Q_A]  ──► reuse Cache(P, D1, D3)
  Prompt B: [P] + {D2} + [Q_B]      ──► reuse Cache(P, D2)

Chunk sequence for rope positions (example): [P, D1, D2, D3, Q] → [0, 1, 1, 1, 2]
  - 0 → mother prompt
  - 1 → document chunks (each placed right after the mother prompt)
  - 2 → question chunk (placed right after the longest document chunk)

Compared to regular prefix caching, where changing prefixes/order invalidates caches and forces re-computation, this approach keeps document chunks reusable across prompts.
```

### Install
```bash
pip install "vllm==0.10.2"
```

### Run
```bash
python test.py
```

`test.py` imports and runs `monkey_patch_vllm.patch_vllm()` to apply the runtime patches.

- Change the documents by editing `DOCUMENT_POOL` in `test.py`.

- `monkey_patch_vllm.py` and `patch_gpu_model_runner.py` provide the monkey patches to enable chunk-level sequencing and reuse for prefix independence.

### Mix-up example (prefix confusion)
Run `python test_mixup.py` to observe potential cross-chunk mix-ups.

- Toggle mode by setting `run_type` at the top of `test_mixup.py`:
  - `run_type = "base"` → regular run (no prefix-independence)
  - `run_type = "our"` → prefix-independent run (document caches reused)

- Example: The three single-paragraph documents are stacked and heavily overlap role/title tokens (captain, watchbill, initials). The question asks for the captain’s name of the Kestrel. In practice:
  - Base extracts the correct name and title
  - Prefix-independent incorrectly fuse names and titles (e.g., "E. Sveinsson" is considered to be "receiving officer").

- Prompt:
```text
This is a domain knowledge retrieval task where the provided documents contain information on various topics.
Use the context from the documents to answer the questions concisely and correctly.
Do not add information not found in the documents.

Watchbill extract — Captain — Einar Sveinsson; Quartermaster — R. Almeida; Navigator — Lea Park; Mechanic — Hari Singh;
entries tagged 'capt.' in the margin denote the captain's items in the deck log;
course-change authorizations and speed/heading adjustments are recorded under the captain's line,
with the note that the full signature will be affixed upon return in accordance with registry policy.

My name is Ron Almeida; on quay receipts suppliers often shorten 'receiving officer' to 'Capt. Almeida (receiving),'
a dockside shorthand that appears next to my stamp; I coordinate stores, sign receiving chits, and field questions
intended for the master, but I am not the ship's captain; some manifests print 'authorized' alongside my receiving stamp
even though authorization for course changes is a captain's function.

Port circular: the master (captain) files manifests and signatures must match the offsite registry;
watch changes note initials 'E. Sveinsson' for captain duties while 'R. Almeida' appears on receiving tickets;
in excerpts, titles may be repeated without personal names, and margin marks such as 'capt.' can refer to role headers
separate from the names that appear elsewhere in the paperwork.

What is the name of the captain of the Kestrel?
```

- Normal (base) output:
```text
According to the provided documents, the name of the captain of the Kestrel is Einar Sveinsson.
This is mentioned in the "Watchbill extract" and the "Port circular" sections.
Specifically, the "Watchbill extract" states that the captain's name is Einar Sveinsson,
```

- Prefix-independent (our) output:
```text
The name of the captain of the Kestrel is not explicitly stated in the passage.
However, it is mentioned that the captain's name is not "E. Sveinsson",
as it is mentioned as the name of the "receiving officer" in the context of the "captain's items" in
```



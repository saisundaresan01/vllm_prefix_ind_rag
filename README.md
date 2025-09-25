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
Run `python test_mixup.py` to observe cross-chunk mix-ups.

- Toggle mode by setting `run_type` at the top of `test_mixup.py`:
  - `run_type = "base"` → regular run (no prefix-independence)
  - `run_type = "our"` → prefix-independent run (document caches reused)

#### Generation Results

- Example 1 (Captain of the Kestrel)
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
  - Normal:
    ```text
    According to the provided documents, the name of the captain of the Kestrel is Einar Sveinsson. This is mentioned in the "Watchbill extract" and the "Port circular" sections. Specifically, the "Watchbill extract" states that the captain's name is Einar Sveinsson,
    ```
  - Our:
    ```text
    The name of the captain of the Kestrel is not explicitly stated in the passage. However, it is mentioned that the captain's name is not "E. Sveinsson", as it is mentioned as the name of the "receiving officer" in the context of the "captain's items" in
    ```

- Example 2 (Civic engineering — Riverside Viaduct hours)
  - Prompt:
    ```text
    This is a domain knowledge retrieval task where the provided documents contain information on various topics.
    Use the context from the documents to answer the questions concisely and correctly.
    Do not add information not found in the documents.

    The City Engineering Division will reduce southbound lanes on the Riverside Viaduct this Saturday from 07:00 to 11:00 for joint inspection and deck sounding. Flaggers will be present at both approaches, and bicycle access will remain open on the east walkway. Northbound lanes are unaffected. The bulletin clarifies that this is a limited, daytime lane reduction and not an overnight shutdown, and that all listed times are in local time as shown on municipal notices.

    Separately, the Airport Authority will close the Runway Access Bridge overnight from 22:00 to 06:00 next week to complete resurfacing of expansion joints. Detours will route traffic along Harbor Road with signage indicating Riverside Viaduct as an alternate daytime corridor; however, the viaduct will not be subject to the overnight closure window. The notice uses airport operations schedules and refers only to the Runway Access Bridge, which is distinct from the Riverside Viaduct in jurisdiction and timing.

    A weekend daylight‑saving time advisory notes that some private traveler apps may display one‑hour offsets until caches refresh on Monday. City bulletins, including the Riverside Viaduct lane reduction, list times in the local civil clock and are unaffected by airline schedule conventions. Residents are advised to consult the municipal website for the most accurate local start and end times when planning around construction or inspection activities.

    On Saturday, during what hours are the southbound lanes of the Riverside Viaduct reduced?
    ```
  - Normal:
    ```text
    According to the document, the southbound lanes of the Riverside Viaduct will be reduced from 07:00 to 11:00 on Saturday. This is a limited, daytime lane reduction, and not an overnight shutdown. The listed times are in local time, as shown on municipal notices. Therefore, the answer
    ```
  - Our:
    ```text
    The southbound lanes of the Riverside Viaduct are not reduced. The notice only mentions the Runway Access Bridge, not the Riverside Viaduct. The Runway Access Bridge will have reduced southbound lanes from 07:00 to 11:00 on Saturday. The Riverside Viaduct is not affected.
    ```

- Example 3 (Procurement tender — Contract RIV‑842)
  - Prompt:
    ```text
    This is a domain knowledge retrieval task where the provided documents contain information on various topics.
    Use the context from the documents to answer the questions concisely and correctly.
    Do not add information not found in the documents.

    Public Notice — City of Eastport invites sealed bids for Contract RIV‑842 (Riverside Viaduct deck sounding and joint inspection) and sister Contract RIV‑842A (airport branch survey). Bids due June 12 at 10:00. Read‑aloud prices (subject to verification): ACW $1.42M, VDE $1.47M, RDS $1.51M for RIV‑842. Apparent low at opening may not be final; standings may change after arithmetic corrections and addendum acknowledgments. A draft NOI template (prepared for 842A) erroneously carries a mail‑merge header; such text is not binding and will be superseded by compliance findings; the lowest responsive and responsible bidder is 

    A. Chen Works (ACW). On the read‑aloud list for RIV‑842; however, arithmetic correction on ACW Schedule B (+$110k) adjusts ACW to $1.53M for RIV‑842; VDE adjusts to $1.49M after Addendum 2 quantity change. RDS failed to acknowledge Addendum 2 on the signature page. License renewal for VDE timestamped 08:02 on June 12 (before 10:00 opening). Preliminary DBE reports: VDE 12.2%, ACW 11.9%, RDS 13.0% (subject to Compliance). Committee notes the prior mail‑merge header was a template artifact; no award decision is taken here; the lowest responsive and responsible bidder is

    ACW — Compliance & Legal Memo — RIV‑842/842A (June 18): Award shall be made to the lowest responsive and responsible bidder meeting all requirements, including acknowledgment of latest addenda and meeting the DBE goal of 12.0%. Findings: RDS is ineligible due to active 90‑day debarment (APR‑511, eff. Jun 5). ACW’s non‑collusion affidavit has a notary commission mismatch and ACW’s DBE commitment (11.9%) is below the 12.0% goal; corrective affidavit received June 16 is after the deadline. VDE acknowledged Addendum 2 and has timely license renewal; DBE prelim 12.2% (meets goal). Any previously issued NOI that conflicts with these findings is rescinded. No single vendor is named here as awardee; eligibility determinations are returned to the record for application of price rankings.

    According to the procurement rules and the final applicability of corrections and eligibility, who receives Contract RIV-842?
    ```
  - Normal:
    ```text
    According to the documents, VDE (VDE $1.47M) is the lowest responsive and responsible bidder meeting all requirements, including acknowledgment of latest addenda and meeting the DBE goal of 12.0%. Therefore, VDE receives Contract RIV-842. Note that ACW's DBE commitment
    ```
  - Our:
    ```text
    Based on the provided documents, the answer is: The lowest responsive and responsible bidder.

    The document states: "The lowest responsive and responsible bidder is the lowest bidder that meets all the requirements and is deemed eligible." This is stated in the sentence: "The lowest responsive and responsible bidder is 12.2%."
    ```

Notes:
- In Example 1, the three single-paragraph documents are stacked and heavily overlap role/title tokens (captain, watchbill, initials). The question asks for the captain’s name of the Kestrel. Base extracts the correct name and title, while our method incorrectly fuse names and titles (e.g., "E. Sveinsson" is considered to be "receiving officer").
- In Example 2, placing "Runway Access Bridge" tokens adjacent to the viaduct's "07:00–11:00" window under our method induces cross-chunk mixup that flips the answer.
- In Example 3, award language fragments (“the lowest responsive and responsible bidder is …”) appear near the ACW name across chunk boundaries; the compliance text disqualifies ACW later. Under our method, the model can get confused and assigns ACW to the award phrase.



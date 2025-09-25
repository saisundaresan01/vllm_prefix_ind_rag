CHUNK_RECOMPUTE_FLAGS = {}

enforce_eager = True
max_tokens = 64
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tp = 1
run_type = "our"
file_path = f"test_outputs/output_{run_type}.csv"

if run_type == "our":
    from monkey_patch_vllm import patch_vllm
    patch_vllm()  # Apply monkey patches safely

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

DOCUMENT_POOL = {
    "case_dog_diary": " Watchbill extract — Captain — Einar Sveinsson; Quartermaster — R. Almeida; Navigator — Lea Park; Mechanic — Hari Singh; entries tagged 'capt.' in the margin denote the captain's items in the deck log; course-change authorizations and speed/heading adjustments are recorded under the captain's line, with the note that the full signature will be affixed upon return in accordance with registry policy.",
    "case_ron_memoir": " My name is Ron Almeida; on quay receipts suppliers often shorten 'receiving officer' to 'Capt. Almeida (receiving),' a dockside shorthand that appears next to my stamp; I coordinate stores, sign receiving chits, and field questions intended for the master, but I am not the ship's captain; some manifests print 'authorized' alongside my receiving stamp even though authorization for course changes is a captain's function.",
    "case_clinic_bulletin": " Port circular: the master (captain) files manifests and signatures must match the offsite registry; watch changes note initials 'E. Sveinsson' for captain duties while 'R. Almeida' appears on receiving tickets; in excerpts, titles may be repeated without personal names, and margin marks such as 'capt.' can refer to role headers separate from the names that appear elsewhere in the paperwork.",
    "tender_public_notice": " Public Notice — City of Eastport invites sealed bids for Contract RIV‑842 (Riverside Viaduct deck sounding and joint inspection) and sister Contract RIV‑842A (airport branch survey). Bids due June 12 at 10:00. Read‑aloud prices (subject to verification): ACW $1.42M, VDE $1.47M, RDS $1.51M for RIV‑842. Apparent low at opening may not be final; standings may change after arithmetic corrections and addendum acknowledgments. A draft NOI template (prepared for 842A) erroneously carries a mail‑merge header; such text is not binding and will be superseded by compliance findings; the lowest responsive and responsible bidder is ",
    "tender_committee_minutes": " A. Chen Works (ACW). On the read‑aloud list for RIV‑842; however, arithmetic correction on ACW Schedule B (+$110k) adjusts ACW to $1.53M for RIV‑842; VDE adjusts to $1.49M after Addendum 2 quantity change. RDS failed to acknowledge Addendum 2 on the signature page. License renewal for VDE timestamped 08:02 on June 12 (before 10:00 opening). Preliminary DBE reports: VDE 12.2%, ACW 11.9%, RDS 13.0% (subject to Compliance). Committee notes the prior mail‑merge header was a template artifact; no award decision is taken here; the lowest responsive and responsible bidder is",
    "tender_compliance_memo": " ACW — Compliance & Legal Memo — RIV‑842/842A (June 18): Award shall be made to the lowest responsive and responsible bidder meeting all requirements, including acknowledgment of latest addenda and meeting the DBE goal of 12.0%. Findings: RDS is ineligible due to active 90‑day debarment (APR‑511, eff. Jun 5). ACW’s non‑collusion affidavit has a notary commission mismatch and ACW’s DBE commitment (11.9%) is below the 12.0% goal; corrective affidavit received June 16 is after the deadline. VDE acknowledged Addendum 2 and has timely license renewal; DBE prelim 12.2% (meets goal). Any previously issued NOI that conflicts with these findings is rescinded. No single vendor is named here as awardee; eligibility determinations are returned to the record for application of price rankings.",
    "civic_bulletin_viaduct": " The City Engineering Division will reduce southbound lanes on the Riverside Viaduct this Saturday from 07:00 to 11:00 for joint inspection and deck sounding. Flaggers will be present at both approaches, and bicycle access will remain open on the east walkway. Northbound lanes are unaffected. The bulletin clarifies that this is a limited, daytime lane reduction and not an overnight shutdown, and that all listed times are in local time as shown on municipal notices.",
    "civic_airport_bridge": " Separately, the Airport Authority will close the Runway Access Bridge overnight from 22:00 to 06:00 next week to complete resurfacing of expansion joints. Detours will route traffic along Harbor Road with signage indicating Riverside Viaduct as an alternate daytime corridor; however, the viaduct will not be subject to the overnight closure window. The notice uses airport operations schedules and refers only to the Runway Access Bridge, which is distinct from the Riverside Viaduct in jurisdiction and timing.",
    "civic_time_advisory": " A weekend daylight‑saving time advisory notes that some private traveler apps may display one‑hour offsets until caches refresh on Monday. City bulletins, including the Riverside Viaduct lane reduction, list times in the local civil clock and are unaffected by airline schedule conventions. Residents are advised to consult the municipal website for the most accurate local start and end times when planning around construction or inspection activities.",
}

RAG_PROMPTS = [
    {
        "index": 0,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [],
        "question": ""
    },
    {
        "index": 1,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["case_dog_diary"]],
        "question": ""
    },
    {
        "index": 2,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["case_ron_memoir"]],
        "question": ""
    },
    {
        "index": 3,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["case_clinic_bulletin"]],
        "question": ""
    },
    {
        "index": 4,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [
            DOCUMENT_POOL["case_dog_diary"],
            DOCUMENT_POOL["case_ron_memoir"],
            DOCUMENT_POOL["case_clinic_bulletin"],
        ],
        "question": " What is the name of the captain of the Kestrel?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    # Scenario 2: Procurement tender — requires multi-hop reasoning across overlapping paragraphs
    {
        "index": 5,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["tender_public_notice"] + "\n\n" + DOCUMENT_POOL["tender_committee_minutes"]],
        "question": ""
    },
    {
        "index": 6,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["tender_compliance_memo"]],
        "question": ""
    },
    {
        "index": 7,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["tender_committee_minutes"]],
        "question": ""
    },
    {
        "index": 8,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [
            DOCUMENT_POOL["tender_public_notice"],
            DOCUMENT_POOL["tender_committee_minutes"],
            DOCUMENT_POOL["tender_compliance_memo"],
        ],
        "question": " According to the procurement rules and the final applicability of corrections and eligibility, who receives Contract RIV-842?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    # Scenario 3: Civic engineering — can wrongly use an overnight closure or DST offset
    {
        "index": 9,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["civic_bulletin_viaduct"]],
        "question": ""
    },
    {
        "index": 10,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["civic_airport_bridge"]],
        "question": ""
    },
    {
        "index": 11,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["civic_time_advisory"]],
        "question": ""
    },
    {
        "index": 12,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [
            DOCUMENT_POOL["civic_bulletin_viaduct"],
            DOCUMENT_POOL["civic_airport_bridge"],
            DOCUMENT_POOL["civic_time_advisory"],
        ],
        "question": " On Saturday, during what hours are the southbound lanes of the Riverside Viaduct reduced?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    }
]

def print_num_tokens(rag_prompt, tokenizer):
    print("\n")
    concat_elem = ""
    all_elem_len = 0
    num_token_list = []
    for key, elem in rag_prompt.items():
        if isinstance(elem, int):
            print(f"{key}: {elem}")
        elif isinstance(elem, list):
            for i, sub_elem in enumerate(elem): 
                sub_elem_len = len(tokenizer(sub_elem)['input_ids'])-1
                print(f"{key}_{i}: {sub_elem_len}")
                num_token_list.append(sub_elem_len)
                concat_elem += sub_elem
                all_elem_len += sub_elem_len
        elif isinstance(elem, str):
            elem_len  = len(tokenizer(elem)['input_ids'])-1
            if key == "mother_prompt": elem_len += 1
            print(f"{key}: {elem_len}")
            num_token_list.append(elem_len)
            concat_elem += elem
            all_elem_len += elem_len
        else:
            raise ValueError(f"Invalid type for element {key}: {type(elem)}")
        
    # check if the sum of length of all elements is equal to the sum of the concatenated element lengths
    concat_elem_len = len(tokenizer(concat_elem)['input_ids'])
    assert all_elem_len == concat_elem_len, f"Sum of lengths of all elements: {all_elem_len} != Length of concatenated elements: {concat_elem_len}"

    return num_token_list

def format_rag_prompt(rag_prompt, type='base'):
    if type == 'base':
        return rag_prompt["mother_prompt"] + "".join(rag_prompt["documents"]) + rag_prompt["question"]

    if type == 'our':
        chunk_texts = [rag_prompt["mother_prompt"]] + rag_prompt["documents"] + ([] if rag_prompt["question"] == "" else [rag_prompt["question"]])
        chunk_sequence = [0] + [1 for _ in range(len(rag_prompt["documents"]))] + ([] if rag_prompt["question"] == "" else [2])
        return {"chunk_texts": chunk_texts, "chunk_sequence": chunk_sequence}

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare prompts
num_token_lists = [print_num_tokens(rp, tokenizer) for rp in RAG_PROMPTS]
prompts = [format_rag_prompt(rp, type=run_type) for rp in RAG_PROMPTS]

sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, min_tokens=max_tokens)

llm = LLM(model=model_path, max_num_batched_tokens=100000,
            dtype="float16", tensor_parallel_size=tp, enforce_eager=True)


outputs = []
for i,p in enumerate(prompts):
    output = llm.generate(p, sampling_params)
    if RAG_PROMPTS[i]["question"] != "":
        outputs.extend(output)

for output in outputs:
    print("----")
    print(f"Prompt: {output.prompt!r}\n")
    print(f"Generated text: {output.outputs[0].text!r}\n")



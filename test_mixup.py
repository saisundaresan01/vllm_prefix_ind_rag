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
outputs.extend(llm.generate(prompts[0], sampling_params))
outputs.extend(llm.generate(prompts[1], sampling_params))
outputs.extend(llm.generate(prompts[2], sampling_params))
outputs.extend(llm.generate(prompts[3], sampling_params))
outputs.extend(llm.generate(prompts[4], sampling_params))

for output in outputs:
    print("----")
    print(f"Prompt: {output.prompt!r}\n")
    print(f"Generated text: {output.outputs[0].text!r}\n")



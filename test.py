CHUNK_RECOMPUTE_FLAGS = {}

enforce_eager = True
max_tokens = 49
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
    "holy_roman": " The Holy Roman Empire began in 27 BCE when Augustus became its first emperor. It reached its height around 117 CE, spanning three continents. The empire's strength lay in its military, infrastructure, and governance, but it ultimately declined due to political instability, economic troubles, and invasions by barbarian tribes.",
    "photosynthesis": " Photosynthesis is a process which plants use sunlight to synthesize food and is essential to life. Chlorophyll captures light energy, which drives the transformation. The chemical equation for photosynthesis is 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂.",
    "renaissance": " The Renaissance was a movement that started in Italy during the 14th century. It marked the transition from the Middle Ages to modernity. The Renaissance saw advancements in art, science, and literature, with figures like Leonardo da Vinci and Michelangelo contributing significantly. It emphasized humanism, focusing on human potential and achievements."
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
        "documents": [DOCUMENT_POOL["renaissance"]],
        "question": ""
    },
    {
        "index": 2,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["photosynthesis"]],
        "question": ""
    },
    {
        "index": 3,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["holy_roman"]],
        "question": ""
    },
    {
        "index": 4,
        "mother_prompt": "<|start_header_id|>user<|end_header_id|>\n\n This is a domain knowledge retrieval task where the provided documents contain information on various topics. Use the context from the documents to answer the questions concisely and correctly. Do not add information not found in the documents.",
        "documents": [DOCUMENT_POOL["holy_roman"], DOCUMENT_POOL["renaissance"], DOCUMENT_POOL["photosynthesis"]],
        "question": " Tell me when did the Renaissance movement start in Italy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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

def set_chunk_recompute_flags(prompt, tokenizer):
    chunk_texts = prompt["chunk_texts"]
    chunk_recompute_flags = []
    for i, chunk_text in enumerate(chunk_texts):
        if chunk_text not in CHUNK_RECOMPUTE_FLAGS:
            num_tokens = len(tokenizer(chunk_text)['input_ids']) - 1
            if i == 0:
                num_tokens += 1
            CHUNK_RECOMPUTE_FLAGS[chunk_text] = [0] * num_tokens
        chunk_recompute_flags.append(CHUNK_RECOMPUTE_FLAGS[chunk_text])
    prompt["chunk_recompute_flags"] = chunk_recompute_flags
    return prompt


def update_chunk_recompute_flags(chunk_texts, chunk_recompute_flags):
    for i, chunk_text in enumerate(chunk_texts):
        recompute_ratio = 0
        recompute_tokens = int(len(chunk_recompute_flags[i]) * recompute_ratio)
        CHUNK_RECOMPUTE_FLAGS[chunk_text] = (
            [-1] * (len(chunk_recompute_flags[i]) - recompute_tokens) +
            [1] * recompute_tokens
        )

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare prompts
num_token_lists = [print_num_tokens(rp, tokenizer) for rp in RAG_PROMPTS]
prompts = [format_rag_prompt(rp, type=run_type) for rp in RAG_PROMPTS]

sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, min_tokens=max_tokens)

llm = LLM(model=model_path, max_num_batched_tokens=100000,
            dtype="float16", tensor_parallel_size=tp, enforce_eager=True)


outputs = []
prompts[0] = set_chunk_recompute_flags(prompts[0], tokenizer)
outputs.extend(llm.generate(prompts[0], sampling_params))
update_chunk_recompute_flags(prompts[0]["chunk_texts"], prompts[0]["chunk_recompute_flags"])

prompts[1] = set_chunk_recompute_flags(prompts[1], tokenizer)
outputs.extend(llm.generate(prompts[1], sampling_params))
update_chunk_recompute_flags(prompts[1]["chunk_texts"], prompts[1]["chunk_recompute_flags"])

prompts[2] = set_chunk_recompute_flags(prompts[2], tokenizer)
outputs.extend(llm.generate(prompts[2], sampling_params))
update_chunk_recompute_flags(prompts[2]["chunk_texts"], prompts[2]["chunk_recompute_flags"])

prompts[3] = set_chunk_recompute_flags(prompts[3], tokenizer)
outputs.extend(llm.generate(prompts[3], sampling_params))
update_chunk_recompute_flags(prompts[3]["chunk_texts"], prompts[3]["chunk_recompute_flags"])

prompts[4] = set_chunk_recompute_flags(prompts[4], tokenizer)
outputs.extend(llm.generate(prompts[4], sampling_params))
update_chunk_recompute_flags(prompts[4]["chunk_texts"], prompts[4]["chunk_recompute_flags"])

for output in outputs:
    print("----")
    print(f"Prompt: {output.prompt!r}\n")
    print(f"Generated text: {output.outputs[0].text!r}\n")
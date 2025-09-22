import os
import shutil
import atexit
import importlib.util

# --- Paths ---
orig_path = importlib.util.find_spec("vllm.v1.worker.gpu_model_runner").origin
backup_path = "./gpu_model_runner.py"
patched_path = "./patch_gpu_model_runner.py"

shutil.copy2(orig_path, backup_path)

shutil.copy2(patched_path, orig_path)

def restore_original():
    if os.path.exists(backup_path):
        shutil.move(backup_path, orig_path)

atexit.register(restore_original)

from vllm.inputs.data import SingletonPrompt
from vllm.inputs.parse import ParsedSingletonPrompt, ParsedStrPrompt, ParsedEmbedsPrompt, ParsedTokensPrompt, ParsedTextPrompt

import importlib
import sys

from typing import Literal, TypedDict, Any, Optional, Union, Mapping, Callable, NewType
from vllm.lora.request import LoRARequest

from vllm.inputs.data import (SingletonInputs, TextPrompt)
from typing_extensions import assert_never

from vllm.logger import init_logger

from vllm.multimodal.inputs import MultiModalUUIDDict
from vllm.inputs import ProcessorInputs, PromptType
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions

from vllm.inputs.parse import parse_singleton_prompt

from vllm.v1.request import Request, RequestStatus
from vllm.v1.engine import EngineCoreEvent
from vllm.v1.utils import ConstantList
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys, hash_block_tokens

import time
import msgspec
from functools import partial
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.core.sched.output import NewRequestData

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from vllm import bc_linter_include

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

import torch
import numpy as np
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.sample.logits_processor import (BatchUpdateBuilder,
                                             LogitsProcessors)

def InputBatch__init__our(
    self,
    max_num_reqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    device: torch.device,
    pin_memory: bool,
    vocab_size: int,
    block_sizes: list[int],  # The block_size of each kv cache group
    logitsprocs: Optional[LogitsProcessors] = None,
    is_spec_decode: bool = False,
    is_pooling_model: bool = False,
    num_speculative_tokens: int = 0,
):
    self.is_pooling_model = is_pooling_model
    self.is_spec_decode = is_spec_decode
    self.max_num_reqs = max_num_reqs
    self.max_model_len = max_model_len
    self.max_num_batched_tokens = max_num_batched_tokens
    self.device = device
    self.pin_memory = pin_memory
    self.vocab_size = vocab_size

    self._req_ids: list[Optional[str]] = []
    self.req_id_to_index: dict[str, int] = {}

    # TODO(woosuk): This buffer could be too large if max_model_len is big.
    # Find a way to reduce the CPU memory usage.
    # This buffer is not directly transferred to the GPU, so it does not
    # need to be pinned.
    self.token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
        pin_memory=False,
    )
    self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
    self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
    self.num_computed_tokens_cpu_tensor = torch.zeros(
        (max_num_reqs, ),
        device="cpu",
        dtype=torch.int32,
        pin_memory=pin_memory,
    )

    self.max_prefix_ind_positions_cpu_tensor = torch.zeros(
        (max_num_reqs, ),
        device="cpu",
        dtype=torch.int32,
        pin_memory=pin_memory,
    )
    
    self.num_computed_tokens_cpu = \
        self.num_computed_tokens_cpu_tensor.numpy()

    self.max_prefix_ind_positions_cpu = \
        self.max_prefix_ind_positions_cpu_tensor.numpy()

    # Block table.
    self.block_table = MultiGroupBlockTable(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=pin_memory,
        device=device,
        block_sizes=block_sizes,
        num_speculative_tokens=num_speculative_tokens,
    )

    # Sampling-related.
    self.temperature = torch.empty((max_num_reqs, ),
                                    dtype=torch.float32,
                                    device=device)
    self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                dtype=torch.float32,
                                                device="cpu",
                                                pin_memory=pin_memory)
    self.temperature_cpu = self.temperature_cpu_tensor.numpy()
    self.greedy_reqs: set[str] = set()
    self.random_reqs: set[str] = set()

    self.top_p = torch.empty((max_num_reqs, ),
                                dtype=torch.float32,
                                device=device)
    self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                        dtype=torch.float32,
                                        device="cpu",
                                        pin_memory=pin_memory)
    self.top_p_cpu = self.top_p_cpu_tensor.numpy()
    self.top_p_reqs: set[str] = set()

    self.top_k = torch.empty((max_num_reqs, ),
                                dtype=torch.int32,
                                device=device)
    self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=pin_memory)
    self.top_k_cpu = self.top_k_cpu_tensor.numpy()
    self.top_k_reqs: set[str] = set()

    # IDs of requests which do not support spec decoding
    self.spec_decode_unsupported_reqs: set[str] = set()

    # Frequency penalty related data structures
    self.frequency_penalties = torch.empty((max_num_reqs, ),
                                            dtype=torch.float,
                                            device=device)
    self.frequency_penalties_cpu_tensor = torch.empty(
        (max_num_reqs, ),
        dtype=torch.float,
        device="cpu",
        pin_memory=pin_memory)
    self.frequency_penalties_cpu = \
        self.frequency_penalties_cpu_tensor.numpy()
    self.frequency_penalties_reqs: set[str] = set()

    # Presence penalty related data structures
    self.presence_penalties = torch.empty((max_num_reqs, ),
                                            dtype=torch.float,
                                            device=device)
    self.presence_penalties_cpu_tensor = torch.empty((max_num_reqs, ),
                                                        dtype=torch.float,
                                                        device="cpu",
                                                        pin_memory=pin_memory)
    self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy(
    )
    self.presence_penalties_reqs: set[str] = set()

    # Repetition penalty related data structures
    self.repetition_penalties = torch.empty((max_num_reqs, ),
                                            dtype=torch.float,
                                            device=device)
    self.repetition_penalties_cpu_tensor = torch.empty(
        (max_num_reqs, ),
        dtype=torch.float,
        device="cpu",
        pin_memory=pin_memory)
    self.repetition_penalties_cpu = \
        self.repetition_penalties_cpu_tensor.numpy()
    self.repetition_penalties_reqs: set[str] = set()

    # Speculative decoding
    self.num_accepted_tokens_cpu_tensor = torch.ones((max_num_reqs, ),
                                                        dtype=torch.int64,
                                                        device="cpu",
                                                        pin_memory=pin_memory)
    self.num_accepted_tokens_cpu = \
        self.num_accepted_tokens_cpu_tensor.numpy()

    # lora related
    self.request_lora_mapping = np.zeros((self.max_num_reqs, ),
                                            dtype=np.int32)
    self.lora_id_to_request_ids: dict[int, set[str]] = {}
    self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

    # req_index -> generator
    # NOTE(woosuk): The indices of the requests that do not have their own
    # generator should not be included in the dictionary.
    self.generators: dict[int, torch.Generator] = {}

    self.num_logprobs: dict[str, int] = {}
    # NOTE(rob): num_prompt_logprobs only includes reqs
    # that are currently in the prefill phase.
    self.num_prompt_logprobs: dict[str, int] = {}

    # To accumulate prompt logprobs tensor chunks across prefill steps.
    self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

    # Internal representation of per-step batch state changes, used for
    # reordering persistent batch and generating logitsprocs batch state
    # updates. Should reset each step.
    self.batch_update_builder = BatchUpdateBuilder()

    # TODO convert this to LogitsProcessor
    self.has_allowed_token_ids: set[str] = set()
    # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
    # the value is False. Since we use masked_fill_ to set -inf.
    self.allowed_token_ids_mask: Optional[torch.Tensor] = None
    self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

    # req_index -> bad_words_token_ids
    self.bad_words_token_ids: dict[int, list[list[int]]] = {}

    self.logits_processing_needs_token_ids = np.zeros(max_num_reqs,
                                                        dtype=bool)

    self.req_output_token_ids: list[Optional[list[int]]] = []

    # Store provided logitsprocs. If none are provided, initialize empty
    # data structure
    self.logitsprocs = logitsprocs or LogitsProcessors()

    # This is updated each time the batch constituents change.
    self.sampling_metadata = self._make_sampling_metadata()

    self.pooling_params: dict[str, PoolingParams] = {}

    # Cached reference to the GPU tensor of previously sampled tokens
    self.prev_sampled_token_ids: Optional[torch.Tensor] = None
    self.prev_sampled_token_ids_invalid_indices: Optional[set[int]] = None
    self.prev_req_id_to_index: Optional[dict[str, int]] = None

@bc_linter_include
@dataclass
class NewRequestData_our:

    req_id: str
    prompt_token_ids: list[int]
    prompt_recompute_flags: list[int]
    chunk_lens: list[int]
    padded_chunk_lens: list[int]
    chunk_sequence: list[int]
    mm_kwargs: list[MultiModalKwargsItem]
    mm_hashes: list[str]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_recompute_flags=request.prompt_recompute_flags,
            chunk_lens=request.chunk_lens,
            padded_chunk_lens=request.padded_chunk_lens,
            chunk_sequence=request.chunk_sequence,
            mm_kwargs=request.mm_kwargs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
        )

    def __repr__(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids={self.prompt_token_ids},"
                f"prompt_recompute_flags={self.prompt_recompute_flags},"
                f"chunk_lens={self.chunk_lens},"
                f"padded_chunk_lens={self.padded_chunk_lens},"
                f"chunk_sequence={self.chunk_sequence},"
                f"mm_kwargs={self.mm_kwargs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request}"
                ")")

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids_len={len(self.prompt_token_ids)},"
                f"mm_kwargs={self.mm_kwargs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request}"
                ")")

def request__init__our(
    self,
    request_id: str,
    prompt_token_ids: list[int],
    chunk_sequence: Optional[list[int]],
    prompt_recompute_flags: Optional[list[int]],
    chunk_lens: Optional[list[int]],
    padded_chunk_lens: Optional[list[int]],
    sampling_params: Optional[SamplingParams],
    pooling_params: Optional[PoolingParams],
    eos_token_id: Optional[int],
    client_index: int = 0,
    arrival_time: Optional[float] = None,
    mm_features: Optional[list[MultiModalFeatureSpec]] = None,
    lora_request: Optional["LoRARequest"] = None,
    structured_output_request: Optional["StructuredOutputRequest"] = None,
    cache_salt: Optional[str] = None,
    priority: int = 0,
    trace_headers: Optional[Mapping[str, str]] = None,
    block_hasher: Optional[Callable[["Request"],
                                    list["BlockHash"]]] = None,
) -> None:
    self.request_id = request_id
    self.client_index = client_index
    self.priority = priority
    self.sampling_params = sampling_params
    self.pooling_params = pooling_params
    # Because of LoRA, the eos token id can be different for each request.
    self.eos_token_id = eos_token_id
    self.lora_request = lora_request
    self.structured_output_request = structured_output_request
    self.arrival_time = arrival_time if arrival_time is not None else \
        time.time()

    self.status = RequestStatus.WAITING
    self.use_structured_output = False
    self.events: list[EngineCoreEvent] = []
    self.stop_reason: Union[int, str, None] = None

    # P/D: Connector-specific KV transfer parameters.
    self.kv_transfer_params: Optional[dict[str, Any]] = None

    if pooling_params is not None:
        # Pooling models.
        self.max_tokens = 1
    elif sampling_params is not None:
        # Generative models.
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens
        if sampling_params.guided_decoding is not None:
            self.status = RequestStatus.WAITING_FOR_FSM
            self.use_structured_output = True

        if sampling_params.extra_args is not None:
            self.kv_transfer_params = \
                sampling_params.extra_args.get("kv_transfer_params")
    else:
        raise ValueError(
            "sampling_params and pooling_params can't both be unset")

    self.prompt_token_ids = prompt_token_ids
    self.chunk_sequence = chunk_sequence
    self.prompt_recompute_flags = prompt_recompute_flags
    self.chunk_lens = chunk_lens
    self.padded_chunk_lens = padded_chunk_lens
    self.num_prompt_tokens = len(self.prompt_token_ids)
    self._output_token_ids: list[int] = []
    self._all_token_ids: list[int] = self.prompt_token_ids.copy()
    self.num_output_placeholders = 0  # Used in async scheduling.
    self.spec_token_ids: list[int] = []
    self.num_computed_tokens = 0
    self.cache_salt: Optional[str] = cache_salt

    # Multi-modal related
    self.mm_features = mm_features or []
    self.num_encoder_inputs = len(self.mm_features)
    self.has_encoder_inputs = self.num_encoder_inputs > 0
    # TODO(sfeng33): Remove these legacy fields after clearing out all
    # references in scheduler and model runner
    self.mm_positions = [f.mm_position for f in self.mm_features]
    self.mm_kwargs = [f.data for f in self.mm_features]
    self.mm_hashes = [f.identifier for f in self.mm_features]

    # Read-only views
    # Prevent directly appending to these lists since
    # they should also be updated simultaneously.
    self.output_token_ids = ConstantList(self._output_token_ids)
    self.all_token_ids = ConstantList(self._all_token_ids)
    # trace_headers
    self.trace_headers = trace_headers
    # State
    # The number of tokens with prefix cache hits.
    self.num_cached_tokens = -1

    # The number of NaNs in logits. A value greater than 0
    # indicates that the output is corrupted
    self.num_nans_in_logits = 0

    self.block_hashes: list[BlockHash] = []
    self.get_hash_new_full_blocks: Optional[Callable[
        [], list[BlockHash]]] = None
    if block_hasher is not None:
        self.get_hash_new_full_blocks = partial(block_hasher, self)
        self.block_hashes = self.get_hash_new_full_blocks()

@classmethod
def from_engine_core_request_our(
    cls, request: EngineCoreRequest,
    block_hasher: Optional[Callable[["Request"], list["BlockHash"]]]
) -> "Request":
    return cls(
        request_id=request.request_id,
        client_index=request.client_index,
        prompt_token_ids=request.prompt_token_ids,
        chunk_sequence=request.chunk_sequence,
        prompt_recompute_flags=request.prompt_recompute_flags,
        chunk_lens=request.chunk_lens,
        padded_chunk_lens=request.padded_chunk_lens,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        eos_token_id=request.eos_token_id,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        structured_output_request=StructuredOutputRequest(
            sampling_params=request.sampling_params) \
                if request.sampling_params else None,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=block_hasher,
    )

logger = init_logger(__name__)
BlockHash = NewType("BlockHash", bytes)

def get_request_block_hasher_our(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (request.block_hashes[-1]
                                 if request.block_hashes else None)
        new_block_hashes: list[BlockHash] = []

        cur_chunk_idx = 0
        token_cnt = 0
        padded_chunk_lens = request.padded_chunk_lens

        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx)

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]

            if token_cnt >= padded_chunk_lens[cur_chunk_idx]:
                cur_chunk_idx += 1
                token_cnt = 0
                prev_block_hash_value = None
            token_cnt += len(block_tokens)

            block_hash = hash_block_tokens(caching_hash_fn,
                                           prev_block_hash_value, block_tokens,
                                           extra_keys)

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes
    
    return request_block_hasher

class EngineCoreRequest_our(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    prompt_token_ids: list[int]
    chunk_sequence: Optional[list[int]]
    prompt_recompute_flags: Optional[list[int]]
    chunk_lens: Optional[list[int]]
    padded_chunk_lens: Optional[list[int]]
    mm_features: Optional[list[MultiModalFeatureSpec]]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    cache_salt: Optional[str]
    data_parallel_rank: Optional[int]

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Optional[Mapping[str, str]] = None

def process_inputs_our(
    self,
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    priority: int = 0,
    data_parallel_rank: Optional[int] = None,
) -> tuple[Optional[str], EngineCoreRequest]:

    # TODO(woosuk): Support pooling models.
    self._validate_lora(lora_request)
    self._validate_params(params, lora_request)

    data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
    if data_parallel_rank is not None and not (0 <= data_parallel_rank <
                                                data_parallel_size):
        raise ValueError(f"data_parallel_rank {data_parallel_rank} "
                            f"is out of range [0, {data_parallel_size}).")

    if arrival_time is None:
        arrival_time = time.time()

    # Optionally generate multimodal hash overrides to avoid hashing
    # multimodal data items by their content as their identifiers.

    # NOTE: when users explicitly turn off BOTH prefix caching and input
    # processing caching, no multimodal features or embeddings will be
    # reused across requests, therefore identifying multimodal data items
    # by their content is no longer necessary, and we create uuids with
    # request id-modality-index as multimodal hash overrides.
    if (self.model_config.multimodal_config and
            self.model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.cache_config.enable_prefix_caching):
        mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
    else:
        # Otherwise, use user-provided uuids as multimodal hash overrides
        # if provided.
        self._validate_multi_modal_uuids(prompt)
        if isinstance(prompt, dict):
            mm_uuids = prompt.get("multi_modal_uuids")
        else:
            mm_uuids = None

    # Process inputs, which includes:
    # 1. Tokenize text prompt, with LoRA request if one exists.
    # 2. For multimodal models with a merged preprocessor, preprocess
    #   multimodal data and expand prompt token ids accordingly.
    processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
        prompt,
        tokenization_kwargs=tokenization_kwargs,
        lora_request=lora_request,
        mm_uuids=mm_uuids,
    )
    from vllm.platforms import current_platform
    current_platform.validate_request(
        prompt=prompt,
        params=params,
        processed_inputs=processed_inputs,
    )

    eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

    self._validate_model_inputs(processed_inputs, lora_request)

    encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        # TODO: can we avoid cloning here in multiproc case?
        sampling_params = params.clone()
        # If unset max tokens, then generate up to the max_model_len.
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = (
                self.model_config.max_model_len -
                len(decoder_inputs["prompt_token_ids"]))
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)
        if self.tokenizer is not None:
            sampling_params.update_from_tokenizer(
                self.tokenizer.get_lora_tokenizer(lora_request))
    else:
        pooling_params = params.clone()

    # Multimodal related.
    mm_features: Optional[list[MultiModalFeatureSpec]] = None

    if decoder_inputs["type"] == "multimodal":
        decoder_mm_inputs = decoder_inputs["mm_kwargs"]
        decoder_mm_positions = decoder_inputs["mm_placeholders"]
        decoder_mm_hashes = decoder_inputs["mm_hashes"]

        # Merge and flatten multimodal placeholders, hashes and inputs
        # from dictionaries to lists, and sort them by each item's position
        # in the input sequence.
        sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

        mm_features = []
        for modality, idx in sorted_mm_idxs:
            mm_features.append(
                MultiModalFeatureSpec(
                    data=decoder_mm_inputs[modality][idx],
                    modality=modality,
                    identifier=decoder_mm_hashes[modality][idx],
                    mm_position=decoder_mm_positions[modality][idx]))

    return decoder_inputs.get("prompt"), EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=decoder_inputs["prompt_token_ids"],
        chunk_sequence=decoder_inputs["chunk_sequence"],
        prompt_recompute_flags=decoder_inputs["prompt_recompute_flags"],
        chunk_lens=decoder_inputs["chunk_lens"],
        padded_chunk_lens=decoder_inputs["padded_chunk_lens"],
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=decoder_inputs.get("cache_salt"),
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
    )

# TODO: change to required pad_token_id or mask pad tokens in attention
def get_pad_token_id_our(self,
                    lora_request: Optional[LoRARequest] = None
                ) -> Optional[int]:
    if self.tokenizer is None:
        logger.warning("Using None for BOS token id because tokenizer "
                        "is not initialized")
        return None

    return self.tokenizer.get_lora_tokenizer(lora_request).bos_token_id

def pad_chunks_our(self, block_size: int, pad_token_id: int, chunks: list[list[int]], chunk_recompute_flags: list[int]) -> tuple[list[list[int]], list[int]]:
    """
    Pad a list of chunks to the closest multiple of block size
    """
    padded_chunks = []
    padded_chunk_recompute_flags = []
    for chunk_recompute_flag,chunk in zip(chunk_recompute_flags, chunks):
        num_pad_tokens = (block_size - (len(chunk) % block_size)) % block_size
        chunk = [pad_token_id] * num_pad_tokens + chunk
        chunk_recompute_flag = [2] * num_pad_tokens + chunk_recompute_flag
        padded_chunks.append(chunk)
        padded_chunk_recompute_flags.append(chunk_recompute_flag)
    
    return padded_chunks, padded_chunk_recompute_flags

class ChunkDictInputs(TypedDict):
    """Represents chunked inputs."""

    type: Literal["chunk_dict"]
    """The type of inputs."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    prompt: str
    """The prompt to pass to the model."""

    chunk_sequence: list[int]
    """The sequence of the chunks."""

    prompt_recompute_flags: list[int]
    """The recompute flags of the prompt."""

    chunk_lens: list[int]
    """The lengths of the chunks."""

    padded_chunk_lens: list[int]
    """The padded lengths of the chunks."""

def chunk_dict_inputs_our(
    prompt_token_ids: list[int],
    prompt: str,
    chunk_sequence: list[int],
    prompt_recompute_flags: list[int],
    chunk_lens: list[int],
    padded_chunk_lens: list[int],
) -> ChunkDictInputs:
    inputs = ChunkDictInputs(type="chunk_dict", prompt_token_ids=prompt_token_ids, prompt=prompt, chunk_sequence=chunk_sequence, prompt_recompute_flags=prompt_recompute_flags, chunk_lens=chunk_lens, padded_chunk_lens=padded_chunk_lens)
    return inputs

class ChunkDictPrompt(TypedDict):
    """Schema for a chunked prompt."""

    chunk_texts: list[str]
    """A list of chunks to pass to the model."""

    chunk_sequence: list[int]
    """A list of indices to indicate the order of the chunks."""

    chunk_recompute_flags: list[int]
    """A list of flags to indicate whether to recompute the chunk tokens."""

class ParsedChunkDictPrompt(TypedDict):
    type: Literal["chunk_dict"]
    content: ChunkDictPrompt

def parse_singleton_prompt_our(prompt: SingletonPrompt) -> ParsedSingletonPrompt:
    if isinstance(prompt, str):
        return ParsedStrPrompt(type="str", content=prompt)
    elif isinstance(prompt, dict):
        # Type ignores are because mypy does not correctly infer the TypedDicts
        # Pyright does succeed.
        if "prompt_embeds" in prompt:
            return ParsedEmbedsPrompt(
                type="embeds", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt_token_ids" in prompt:
            return ParsedTokensPrompt(
                type="tokens", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt" in prompt:
            return ParsedTextPrompt(type="text", content=prompt)
        elif "chunk_texts" in prompt:
            return ParsedChunkDictPrompt(type="chunk_dict", content=prompt)
    raise TypeError(
        "inputs must be a string, TextPrompt, TokensPrompt, EmbedsPrompt, or ChunkDictPrompt")

def _process_chunk_dict_our(
    self,
    parsed_content: ChunkDictPrompt,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    lora_request: Optional[LoRARequest] = None,
    block_size: int = 16
) -> ChunkDictInputs:
    chunk_texts = parsed_content["chunk_texts"]
    chunk_sequence = parsed_content["chunk_sequence"]
    prompt = "".join(chunk_texts)

    inputs: ChunkDictInputs
    chunk_token_ids = [self._tokenize_prompt(
        chunk_text,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
    )[1:] for chunk_text in chunk_texts]

    bos_token_id = self.get_bos_token_id(lora_request)
    chunk_token_ids[0] = [bos_token_id] + chunk_token_ids[0] #TODO remove after fixing bos cache logic

    chunk_lens = [len(chunk) for chunk in chunk_token_ids]
    pad_token_id = self.get_pad_token_id(lora_request)

    chunk_recompute_flags = parsed_content.get("chunk_recompute_flags", [[0]*len(ids) for ids in chunk_token_ids])

    padded_chunk_token_ids, padded_chunk_recompute_flags = self.pad_chunks(block_size, pad_token_id, chunk_token_ids, chunk_recompute_flags)
    padded_chunk_lens = [len(chunk) for chunk in padded_chunk_token_ids]

    prompt_token_ids = sum(padded_chunk_token_ids, [])
    prompt_recompute_flag = sum(padded_chunk_recompute_flags, [])

    inputs = chunk_dict_inputs_our(
        prompt_token_ids=prompt_token_ids,
        prompt=prompt,
        chunk_sequence=chunk_sequence,
        prompt_recompute_flags=prompt_recompute_flag,
        chunk_lens=chunk_lens,
        padded_chunk_lens=padded_chunk_lens,
    )

    return inputs

def _prompt_to_llm_inputs_our(
    self,
    prompt: SingletonPrompt,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    lora_request: Optional[LoRARequest] = None,
    *,
    mm_uuids: Optional[MultiModalUUIDDict] = None,
    block_size: int = 16
) -> SingletonInputs:
    """
    Extract the singleton inputs from a prompt.

    Arguments:

    * prompt: single encoder or decoder input prompt
    * lora_request: this is only valid for decoder prompts

    Returns:

    * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
    """
    parsed = parse_singleton_prompt(prompt)

    if parsed["type"] == "embeds":
            return self._process_embeds(parsed["content"])
    if parsed["type"] == "tokens":
        return self._process_tokens(
            parsed["content"],
            lora_request=lora_request,
            mm_uuids=mm_uuids,
        )
    if parsed["type"] == "text":
        return self._process_text(
            parsed["content"],
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            mm_uuids=mm_uuids,
        )
    if parsed["type"] == "str":
        return self._process_text(
            TextPrompt(prompt=parsed["content"]),
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            mm_uuids=mm_uuids,
        )
    if parsed["type"] == "chunk_dict":
        return self._process_chunk_dict(
            parsed["content"],
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            block_size=block_size
        )

    assert_never(parsed)

def patch_vllm():

    parse_module = importlib.import_module("vllm.inputs.parse")
    parse_module.parse_singleton_prompt = parse_singleton_prompt_our

    preprocess_module = importlib.import_module("vllm.inputs.preprocess")
    preprocess_module.InputPreprocessor._prompt_to_llm_inputs = _prompt_to_llm_inputs_our
    preprocess_module.InputPreprocessor.get_pad_token_id = get_pad_token_id_our
    preprocess_module.InputPreprocessor.pad_chunks = pad_chunks_our
    preprocess_module.InputPreprocessor._process_chunk_dict = _process_chunk_dict_our

    engine_module = importlib.import_module("vllm.v1.engine")
    engine_module.EngineCoreRequest = EngineCoreRequest_our

    processor_module = importlib.import_module("vllm.v1.engine.processor")
    processor_module.Processor.process_inputs = process_inputs_our

    core_module = importlib.import_module("vllm.v1.core")
    core_module.kv_cache_utils.get_request_block_hasher = get_request_block_hasher_our

    request_module = importlib.import_module("vllm.v1.request")
    request_module.Request.__init__ = request__init__our
    request_module.Request.from_engine_core_request = from_engine_core_request_our

    sched_output_module = importlib.import_module("vllm.v1.core.sched.output")
    sched_output_module.NewRequestData = NewRequestData_our

    gpu_input_batch_module = importlib.import_module("vllm.v1.worker.gpu_input_batch")
    gpu_input_batch_module.InputBatch.__init__ = InputBatch__init__our

    # runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    # runner_module.GPUModelRunner._get_cumsum_and_arange = _get_cumsum_and_arange_our

    # Rebind already imported symbols
    for name, module in sys.modules.items():
        if module and hasattr(module, "parse_singleton_prompt"):
            module.parse_singleton_prompt = parse_singleton_prompt_our
        if module and getattr(module, "EngineCoreRequest", None) is not None:
            module.EngineCoreRequest = EngineCoreRequest_our
        if name.startswith("vllm") and getattr(module, "Request", None) is not None:
            module.Request.__init__ = request__init__our
            module.Request.from_engine_core_request = from_engine_core_request_our
        if module and getattr(module, "kv_cache_utils", None) is not None:
            module.kv_cache_utils.get_request_block_hasher = get_request_block_hasher_our
        if module and getattr(module, "NewRequestData", None) is not None:
            module.NewRequestData = NewRequestData_our
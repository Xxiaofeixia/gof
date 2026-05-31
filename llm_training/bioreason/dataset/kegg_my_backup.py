import torch
from typing import Any, Dict, List

from functools import partial

from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dna_modules.nucleotide_module import NucleotideDNAModule


def get_format_kegg_function(model_name: str, is_sft: bool=True) -> Any:
    """
    Get the appropriate format function for a given model name.
    """
    if model_name.lower() == "llm":
        return partial(format_kegg_for_llm, is_sft=is_sft)
    elif model_name.lower() == "dna-llm":
        return partial(format_kegg_for_dna_llm, is_sft=is_sft)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
def _format_kegg(example: Dict[str, Any], model_name: str, is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format.
    """

    if model_name.lower() not in ['llm', 'dna-llm']:
        raise ValueError(f"Unsupported model name: {model_name}")

    if model_name.lower() == 'llm':
        question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"
        reference_sequence = ""
        variant_sequence = ""
    elif model_name.lower() == 'dna-llm':
        question = example['question']
        reference_sequence = example['reference_sequence']
        variant_sequence = example['variant_sequence']
    
    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": question.strip()},
                ],
            }
        ],
        "dna_sequences": [
            reference_sequence,
            variant_sequence,
        ],
        "answer": example["answer"],
    }

    if is_sft:
        item['prompt'].append(
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            }
        )
    return item

def format_kegg_for_dna_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for DNA-LLM.
    """
    return _format_kegg(example, 'dna-llm', is_sft=is_sft)

def format_kegg_for_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for LLM.
    """
    return _format_kegg(example, 'llm', is_sft=is_sft)
    

def _truncate_after_assistant_start(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    marker = "<|im_start|>assistant"
    idx = text.find(marker)
    if idx != -1:
        return text[: idx + len(marker)] + "\n"
    return text

def qwen_dna_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length_text: int,
    max_length_dna: int,
    return_answer_in_batch: bool = False,
    truncate_for_generation: bool = True
) -> Dict:
    dna_module = NucleotideDNAModule()
    original_prompts = [example["prompt"] for example in examples]
    
    raw_prompts = dna_module.prepare_prompt(processing_class=processor, inputs=examples)
    
    prompts_text = []
    for msg in raw_prompts:
        if isinstance(msg, list):
            chat_str = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            prompts_text.append(chat_str)
        else:
            prompts_text.append(msg)

    batch_dna_sequences = [example["dna_sequences"] for example in examples]

    batch = processor(
        text=prompts_text,
        batch_dna_sequences=batch_dna_sequences,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
    )
    
    labels = torch.full_like(batch["input_ids"], -100)

    for i, text in enumerate(prompts_text):
        target_str = "<|im_start|>assistant"
        start_idx = text.find(target_str)
        
        if start_idx == -1:
            target_str = "Answer:"
            start_idx = text.find(target_str)

        if start_idx != -1:
            prefix_text = text[:start_idx + len(target_str)]
            prefix_tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_tokens)
            
            pad_count = (batch["input_ids"][i] == processor.tokenizer.pad_token_id).sum().item()
            actual_start_idx = pad_count + prefix_len
            
            seq_len = batch["input_ids"][i].size(0)
            if actual_start_idx < seq_len:
                labels[i, actual_start_idx:] = batch["input_ids"][i, actual_start_idx:]
        else:
            print(f"Warning: No assistant marker found in string! First 200 chars: {text[:200]}")

    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    if return_answer_in_batch:
        batch["answer"] = [example["answer"].strip() for example in examples]

    if truncate_for_generation:
        device = batch["input_ids"].device
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id

        B, L = batch["input_ids"].shape
        keep_lens: List[int] = []

        for i, text in enumerate(prompts_text):
            target_str = "<|im_start|>assistant"
            idx = text.find(target_str)
            if idx == -1:
                target_str = "Answer:"
                idx = text.find(target_str)
            
            if idx != -1:
                prefix_text = text[:idx + len(target_str)] + "\n"
                prefix_tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=False)
                keep_lens.append(len(prefix_tokens))
            else:
                pad_count = (batch["input_ids"][i] == processor.tokenizer.pad_token_id).sum().item()
                keep_lens.append(L - pad_count)

        new_max = max(keep_lens) if keep_lens else 0
        new_input_ids = torch.full((B, new_max), pad_id, dtype=batch["input_ids"].dtype, device=device)
        new_attention = torch.zeros((B, new_max), dtype=batch["attention_mask"].dtype, device=device)
        new_labels = torch.full((B, new_max), -100, dtype=batch["labels"].dtype, device=device)

        for i, k in enumerate(keep_lens):
            if k == 0 or k > L:
                k = L
            src_ids = batch["input_ids"][i, :k]
            src_attn = batch["attention_mask"][i, :k]
            src_lbls = batch["labels"][i, :k]

            new_input_ids[i, -k:] = src_ids
            new_attention[i, -k:] = src_attn
            new_labels[i, -k:] = src_lbls

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention
        batch["labels"] = new_labels

    batch["prompt"] = prompts_text
    batch["original_prompts"] = original_prompts

    final_batch = dict(batch)
    for key, value in final_batch.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            final_batch[key] = torch.tensor(value)
            
    return final_batch
def dna_collate_fn(
    batch: List[Dict[str, Any]],
    dna_tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Custom collate function for DNA models.
    """
    ref_sequences = [item["reference_sequence"] for item in batch]
    alt_sequences = [item["variant_sequence"] for item in batch]

    # Tokenize DNA sequences separately
    tokenized_ref = dna_tokenizer(
        ref_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized_alt = dna_tokenizer(
        alt_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Get labels
    labels = []
    for item in batch:
        label = label2id[item["answer"]]
        labels.append(label)

    # Create labels tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    tokenized_batch = {
        "ref_ids": tokenized_ref.input_ids,
        "ref_attention_mask": tokenized_ref.attention_mask,
        "alt_ids": tokenized_alt.input_ids,
        "alt_attention_mask": tokenized_alt.attention_mask,
        "labels": labels_tensor,
    }

    return tokenized_batch

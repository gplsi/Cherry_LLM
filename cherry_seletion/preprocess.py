import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset
from functools import partial
from typing import List, Dict
from torch import Tensor


def get_template(template_spec, tokenizer):
    if template_spec:
        template_str = None

        # If it's a path, try to read it (as text). If JSON, accept either a raw string or a dict with "chat_template".
        if isinstance(template_spec, str) and os.path.exists(template_spec):
            with open(template_spec, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "chat_template" in parsed:
                    template_str = parsed["chat_template"]
                elif isinstance(parsed, str):
                    template_str = parsed
                else:
                    # Fallback to raw file content if JSON isn't a string or dict with key
                    template_str = content
            except json.JSONDecodeError:
                # Not JSON; treat as a raw Jinja template
                template_str = content
        elif isinstance(template_spec, dict) and "chat_template" in template_spec:
            template_str = template_spec["chat_template"]
        elif isinstance(template_spec, str):
            # Inline template string in config
            template_str = template_spec

        if template_str is not None:
            if not isinstance(template_str, str):
                raise ValueError(
                    "config_model['chat_template'] must be a string, path, or dict with 'chat_template'."
                )
            # Hugging Face expects the template on `tokenizer.chat_template`
            tokenizer.chat_template = template_str


def create_conversation(row):
    conversation = [{"role": "system", "content": row["system"]}]

    for i in range(len(row["input"])):
        conversation.append({"role": "user", "content": row["input"][i]})

        if i < len(row["input"]) - 1:
            conversation.append({"role": "assistant", "content": row["assistance"][i]})

    return conversation


def create_prompt(conversation, tokenizer):
    date_string = datetime.today().strftime("%Y-%m-%d")
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )

    return prompt


def tokenizer_dataset_given_prompt(
    element, tokenizer, max_seq_length, ignore_index=-100
):
    encoded_prompt = torch.tensor(tokenizer.encode(
        element["prompt"], max_length=max_seq_length, add_special_tokens=False
    ),
        dtype=torch.long,
    )
    attention_mask_prompt = torch.ones_like(encoded_prompt)

    encoded_prompt_and_response = tokenizer.encode(
        element["prompt_and_response"],
        max_length=max_seq_length - 1,
        add_special_tokens=False,
    )
    encoded_prompt_and_response.append(tokenizer.eos_token_id)
    encoded_prompt_and_response = torch.tensor(
        encoded_prompt_and_response, dtype=torch.long
    )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_prompt_and_response.clone()

    labels[: len(encoded_prompt)] = ignore_index
    attention_mask = torch.ones_like(encoded_prompt_and_response)

    labels_alone = torch.tensor(
        tokenizer.encode(
            element["response"], max_length=max_seq_length, add_special_tokens=False
        ),
        dtype=torch.long,
    )
    attention_mask_alone = torch.ones_like(labels_alone)

    return {
        "id": torch.tensor(element["id"]).long(),
        "input_ids": encoded_prompt_and_response,
        "labels": labels,
        "attention_mask": attention_mask,
        "labels_alone": labels_alone,
        "attention_mask_alone": attention_mask_alone,
        "prompt_alone": encoded_prompt,
        "attention_mask_prompt": attention_mask_prompt
    }



def tokenizer_dataset_multi_turn(data, tokenizer, max_seq_length=2048) -> Dataset:
    data_result = []

    for element in tqdm(data):
        conversation = create_conversation(element)
        prompt = create_prompt(conversation, tokenizer)
        prompt_and_response = prompt + element["target"] + "<|im_end|>"
        data_result.append(
            {
                "id": element["id"],
                "prompt": prompt,
                "prompt_and_response": prompt_and_response,
                "response": element["target"],
            }
        )

    dataset = Dataset.from_list(data_result)

    dataset = dataset.map(
        tokenizer_dataset_given_prompt,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
    )
    dataset.set_format(
        "torch",
        columns=[
            "id",
            "input_ids",
            "attention_mask",
            "labels",
            "labels_alone",
            "attention_mask_alone",
            "prompt_alone",
            "attention_mask_prompt",
        ],
    )
    return dataset


def get_sft_collate_fn(
    max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(
        _sft_collate_fn,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        ignore_index=ignore_index,
    )


def _sft_collate_fn(
    samples: List[Dict[str, Tensor]],
    max_seq_length: int = -1,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, Tensor]:

    batched = {}
    for key in ("input_ids", "labels", "attention_mask", "labels_alone", "attention_mask_alone", "prompt_alone", "attention_mask_prompt"):
        # Pad values:
        # - input_ids: tokenizer pad token id
        # - labels: ignore_index so loss ignores padding
        # - attention_mask: 0 for padding, 1 for tokens
        if key == "input_ids":
            pad_value = pad_id
        elif key == "labels":
            pad_value = ignore_index
        elif key == "labels_alone":
            pad_value = ignore_index
        elif key == "prompt_alone":
            pad_value = pad_id    
        else:  # attention_mask
            pad_value = 0
        

        # Pad right based on the longest sequence
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples],
            batch_first=True,
            padding_value=pad_value,
        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    return batched

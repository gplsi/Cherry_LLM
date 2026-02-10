import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from preprocess import tokenizer_dataset_multi_turn, get_sft_collate_fn
import time
import warnings
from models import get_model_and_tokenizer

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



def get_perplexity_and_token_losses_cherry(model, input_ids, attention_mask, labels):
    """
    Calculate the perplexity per sequence and token losses
    ONLY over valid tokens (labels != -100).
    Args:
        model: HuggingFace causal LM model
        input_ids: [B, T] tensor
        attention_mask: [B, T] tensor
        labels: [B, T] tensor with -100 in positions to ignore

    Returns:
        perplexity: [B] tensor with perplexity per sequence
        token_losses: [B, T] tensor with loss per token
    """
    with torch.no_grad():
        outputs = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=False
        )

    logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # CE loss per token without reduction
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        logits.view(-1, logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.size())

    # Only consider valid tokens for loss
    valid_mask = (shift_labels != -100) * shift_mask
    token_losses = token_losses * valid_mask

    # PPL per sequence
    lengths = valid_mask.sum(dim=1)
    loss_per_seq = token_losses.sum(dim=1) / valid_mask.sum(dim=1)
    perplexity = torch.exp(loss_per_seq)

    return perplexity.cpu(), token_losses.cpu(), lengths.cpu()




def get_perplexity_and_embedding_whole_text(model, input_ids, attention_mask, labels):
    """
    Calculate the perplexity per sequence and sentence embedding
    ONLY over valid tokens (labels != -100).
    Args:
        model: HuggingFace causal LM model
        input_ids: [B, T] tensor
        attention_mask: [B, T] tensor
        labels: [B, T] tensor with -100 in positions to ignore
    Returns:
        perplexity: [B] tensor with perplexity per sequence
        sentence_embedding: [B, H] tensor with sentence embeddings
    """

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    token_losses = loss_fct(
        logits.view(-1, logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.size())

    # Only consider valid tokens for loss
    valid_mask = (shift_labels != -100) * shift_mask
    token_losses = token_losses * valid_mask

    # PPL per sequence
    loss_per_seq = token_losses.sum(dim=1) / valid_mask.sum(dim=1)
    perplexity = torch.exp(loss_per_seq)

    # ---- Embedding ----
    last_hidden = outputs.hidden_states[-1]
    masked_hidden = last_hidden * attention_mask.unsqueeze(-1)
    sentence_embedding = masked_hidden.sum(dim=1) / attention_mask.sum(
        dim=1, keepdim=True
    )

    return perplexity.cpu(), sentence_embedding.cpu()


def main(args):
    print(args)

    model, tokenizer  = get_model_and_tokenizer(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        chat_template_path=args.chat_template_path,
        device=device,
    )
    model.eval()

    if args.save_path[-3:] != ".pt":
        args.save_path += ".pt"
    if os.path.exists(args.save_path):
        print("save_path exists!")
        raise Exception

    strat_time = time.time()
    new_data = []
    with open(args.data_path, "r") as f:
        json_data = json.load(f)

    dataset = tokenizer_dataset_multi_turn(json_data, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=get_sft_collate_fn(
            max_seq_length=args.max_length, pad_id=0, ignore_index=-100
        ),
    )

    batch_iterator = tqdm(dataloader, mininterval=0, colour="blue")
    for batch in batch_iterator:
        batch = {k: v.to(device) for k, v in batch.items()}

        temp_data_i = {}
        if args.mod == "pre":
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(
                model, batch["prompt_alone"], batch["attention_mask_prompt"], batch["prompt_alone"]
            )
            temp_data_i["ppl"] = [ppl_ins_alone, 0, 0]
            temp_data_i["sent_emb"] = [emb_ins_alone, 0, 0]
            for i in range(ppl_ins_alone.shape[0]):
                new_data.append(
                    {
                        "ppl": [ppl_ins_alone[i].cpu().clone(), 0, 0],
                        "sent_emb": [emb_ins_alone[i].cpu().clone(), 0, 0],
                    }
                )
        elif args.mod == "cherry":
            ppl_out_alone, loss_list_alone, lengths_alone = get_perplexity_and_token_losses_cherry(
                model,
                batch["labels_alone"],
                batch["attention_mask_alone"],
                batch["labels_alone"],
            )
            ppl_out_condition, loss_list_condition, lengths_condition = (
                get_perplexity_and_token_losses_cherry(
                    model, batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
            )

            for i in range(loss_list_alone.shape[0]):
                new_data.append(
                    {
                        "ppl": [
                            0,
                            ppl_out_alone[i].cpu().clone(),
                            ppl_out_condition[i].cpu().clone(),
                        ],
                        "token_loss": [
                            [],
                            loss_list_alone[i].cpu().clone(),
                            loss_list_condition[i].cpu().clone(),
                            lengths_alone[i].cpu().clone(),
                            lengths_condition[i].cpu().clone(),
                        ],
                    }
                )

    print("New data len:", len(new_data))
    torch.save(new_data, args.save_path)

    print("Time Used:", (time.time() - strat_time) / 60, "(min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="/Users/robi/Programacion/Cherry_LLM/data/aya_dataset_es_minor.json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default="/Users/robi/Programacion/Cherry_LLM/data/aya_dataset_es_minor.pt",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--mod", type=str, default="cherry", help="pre, cherry")
    parser.add_argument(
        "--chat_template_path",
        type=str,
        required=False,
        default="/Users/robi/Programacion/Cherry_LLM/chat_template/tokenizer_config_aitana_3_language.json",
    )
    parser.add_argument("--model_type", type=str, required=False, default="llama")
    args = parser.parse_args()
    main(args)

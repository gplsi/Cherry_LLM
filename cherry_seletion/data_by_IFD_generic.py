import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from preprocess import tokenizer_dataset_multi_turn
from models import get_model_and_tokenizer

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def find_subtensor_1d(big, sub):
    n = sub.numel()

    windows = big.unfold(0, n, 1)      # sliding windows
    matches = (windows == sub).all(dim=1)

    idx = matches.nonzero(as_tuple=True)[0]
    return idx.item() if len(idx) else None

def get_loss_part_tokens(input_ids, target_span_tokens, loss_list_, len_loss):
            """
            Split tokens and losses from input_ids and loss_list_ for target_span_tokens.

            Args:
                input_ids: tensor [T] with tokens of input
                target_span_tokens: tensor with tokens of target span to extract losses
                loss_list_: tenseor with losses per token
                len_loss: length expected of loss_list_. To sanitize input. 


            Returns:
                loss_list_clean: np.array with losses only for target span tokens
            """
            # Remove -100 tokens from target_span_tokens
            target_span_tokens_clean = target_span_tokens[target_span_tokens != -100]
            # Remove -100 and padded tokens from loss_list_
            loss_list_without_exclude_token = loss_list_[loss_list_ != -100]
            loss_list_clean = loss_list_without_exclude_token[loss_list_without_exclude_token != 0]

            index = find_subtensor_1d(input_ids, target_span_tokens_clean)
            if index is not None:
                start_token = index
                # Find the index where target_span_tokens appear in input_ids
                span_len = len(target_span_tokens_clean)
                end_token = start_token + span_len
                if len_loss == span_len:
                    tokens_span = input_ids[start_token:end_token]
            else:
                loss_list_clean = None

            if len(tokens_span) != len(loss_list_clean):
                loss_list_clean = None

            return loss_list_clean



def main(args):
    print(args)

    _, tokenizer  = get_model_and_tokenizer(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        device=device,
        chat_template_path=args.chat_template_path,
    )

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    with open(args.data_path, encoding="UTF8") as f:
        json_data = json.load(f)
    dataset = tokenizer_dataset_multi_turn(json_data, tokenizer, args.max_length)

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]
        len_1_list = int(pt_data_i['token_loss'][3])
        len_2_list = int(pt_data_i['token_loss'][4])


        if len_1_list <= 0 or len_2_list <= 0:
            continue
        target_tensor = dataset[i]['labels_alone'][1:]
        

        loss_list_clean_1 = get_loss_part_tokens(dataset[i]['labels_alone'], target_tensor, loss_1_list, len_1_list)
        loss_list_clean_2 = get_loss_part_tokens(dataset[i]['input_ids'], dataset[i]['labels'], loss_2_list, len_2_list)

        mean_1 = loss_list_clean_1.mean()
        mean_2 = loss_list_clean_2.mean()
        mean_rate = mean_2/mean_1
        if mean_rate > 1: 
            continue

        mean_rate_list.append((mean_rate,i))
        mean_list_1.append((mean_1,i))
        mean_list_2.append((mean_2,i))

        # else:
        #     continue

    print('Do Rate')
    mean_rate_list = sorted(mean_rate_list)
    if args.sample_number == 0:
        args.sample_number = int(len(mean_rate_list)*args.sample_rate)
    mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
    mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]
    mean_rate_list_id_sample = sorted(mean_rate_list_id_sample)

    new_data = [json_data[idx] for idx in mean_rate_list_id_sample]
    print('New data len \n',len(new_data))
    with open(args.save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='/Users/robi/Programacion/Cherry_LLM/data/aya_dataset_es_minor.pt')
    parser.add_argument("--data_path", type=str, default='/Users/robi/Programacion/Cherry_LLM/data/aya_dataset_es_minor.json')
    parser.add_argument("--save_path", type=str, default='/Users/robi/Programacion/Cherry_LLM/data/aya_dataset_es_minorcherry.json')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument("--model_type", type=str, required=False, default="llama")
    parser.add_argument(
        "--chat_template_path",
        type=str,
        required=False,
        default="/Users/robi/Programacion/Cherry_LLM/chat_template/tokenizer_config_aitana_3_language.json",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--sample_number", type=int, default=0)
    args = parser.parse_args()
    main(args)
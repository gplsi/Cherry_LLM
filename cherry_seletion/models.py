from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from preprocess import get_template

models_dict_class = {
    "llama": {"model": AutoModelForCausalLM, "tokenizer": AutoTokenizer},
    "Qwen3_VL": {"model": AutoModelForImageTextToText, "tokenizer": AutoProcessor}
}

def get_model_and_tokenizer(model_type, model_name_or_path, chat_template_path, device):
    models_dict = models_dict_class[model_type]
    model = models_dict["model"].from_pretrained(
        model_name_or_path,
        device_map=device,
        output_hidden_states=True,
    )
    tokenizer = models_dict["tokenizer"].from_pretrained(
        model_name_or_path
    )

    if chat_template_path is not None:
        get_template(chat_template_path, tokenizer)
        print("Tokenizer loaded.:", tokenizer.chat_template)
        
    return model, tokenizer
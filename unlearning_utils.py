import torch
import csv
import pandas as pd
import numpy as np
import os
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)


def create_one_hot_and_embeddings(tokens, embed_weights, model):
    if tokens is not None:
        one_hot = create_one_hot(model, tokens)
        embeddings = (one_hot @ embed_weights).data
        return one_hot, embeddings
    else:
        return None, None


def create_one_hot(model, tokens):
    embed_weights = get_embedding_matrix(model)
    if tokens is not None:
        B, sequence_length, dim = tokens.shape[0], tokens.shape[1], embed_weights.shape[0]

        one_hot = torch.zeros(B, sequence_length, dim, device=model.device, dtype=embed_weights.dtype)

        one_hot.scatter_(2, tokens.unsqueeze(2), 1)
        return one_hot
    else:
        return None


def init_attack_embeddings(model, tokenizer, control_prompt, device, repeat=0):
    embed_weights = get_embedding_matrix(model)
    attack_tokens = torch.tensor(tokenizer(control_prompt)["input_ids"], device=device)[1:]
    attack_tokens = attack_tokens.unsqueeze(0)
    if repeat > 0:
        attack_tokens = attack_tokens.repeat(repeat, 1)
    _, embeddings_attack = create_one_hot_and_embeddings(attack_tokens, embed_weights, model)
    embeddings_attack.requires_grad = True
    return embeddings_attack


def print_result_dict(result_dict):
    exclude_from_print = [
        "affirmative_response",
        "embeddings_attack",
        "generated_text",
        "input_tokens",
        "target_tokens",
        "intermediate_layer_generation",
    ]
    str_list = []
    for key, value in result_dict.items():
        if key in exclude_from_print:
            continue
        if isinstance(value, list):
            str_list.append(f"{key}: {value[0]}")
        else:
            str_list.append(f"{key}: {value}")
    final_str = " | ".join(str_list)

    # add generated text if it exists and is not trivial
    if "generated_text" in result_dict:
        generate_text = ""
        if isinstance(result_dict["generated_text"], list):
            if result_dict["generated_text"][0] != None:
                generate_text = f" | generated_text: {result_dict['generated_text'][0]}"
        elif result_dict["generated_text"] != None:
            generate_text = f" | generated_text: {result_dict['generated_text']}"
        if generate_text != "":
            final_str += f"\n==========generated_text============\n{generate_text}\n============================================\n"
    print(final_str)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs
        )
        .to(device)
        .eval()
    )
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = load_tokenizer(tokenizer_path)

    return model, tokenizer


def load_tokenizer(tokenizer_path):
    if "paper_models" in tokenizer_path:
        tokenizer_path = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    set_tokeninzer = 0

    if "llama-2" in tokenizer_path:
        print("Using llama-2 tokenizer, setting padding side to left and pad_token to unk_token.")
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
        set_tokeninzer += 1
    if not tokenizer.pad_token:
        print(
            "Setting pad token to eos token, no specfic logic defined for this model. This might be wrong. Check padding_side and unk_token_id."
        )
        tokenizer.pad_token = tokenizer.eos_token
        set_tokeninzer += 1

    if set_tokeninzer == 0:
        raise ValueError("No tokenizer logic was set. Check logic.")
    if set_tokeninzer > 1:
        raise ValueError("Tokenizer was set more than once. Check logic.")

    return tokenizer


def get_embedding_matrix(model):
    # from llm-attacks
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_attention_mask(model, input_tokens, target_tokens, embeddings_attack):
    B, len_attack = target_tokens.shape[0], embeddings_attack.shape[1]

    attack_mask = torch.ones((B, len_attack), dtype=bool, device=model.device)
    target_mask = target_tokens != 0

    # input_tokens are not used in all datasets
    if input_tokens is not None:
        input_mask = input_tokens != 0
        attention_mask = torch.cat([input_mask, attack_mask, target_mask], dim=1)
    else:
        attention_mask = torch.cat([attack_mask, target_mask], dim=1)

    return attention_mask


def num_affirmative_response(logits_pred, target_tokens, return_sample_wise=False):
    succes_sample_wise = torch.zeros(target_tokens.shape[0], dtype=bool, device=target_tokens.device)
    # for every intermediate layer dimension check success
    for i in range(len(logits_pred)):
        tokens_pred = logits_pred[i].argmax(2)
        success = tokens_pred == target_tokens
        # we do not care about padding tokens and just pretend we predicted them correctly
        success[target_tokens == 0] = True
        succes_sample_wise += success.all(1)

    if return_sample_wise:
        return succes_sample_wise
    else:
        return succes_sample_wise.sum()


DATASETS = ["hp_qa_en"]


def load_dataset_and_dataloader(
    tokenizer, dataset_name, batch_size, csv_columns=[0, 1], test_split=0, shuffle=True, device="cuda:0"
):
    file_path = f"{dataset_name}.csv"
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Choose from {DATASETS}")

    if dataset_name == "harmful_strings":
        csv_columns = [0]  # only contains one column

    reader = csv.reader(open(file_path, "r"))
    cols = next(reader)
    print(f"Using columns: '{[cols[idx] for idx in csv_columns]}' from dataset '{dataset_name}'")
    dataset = list(reader)
    dataset_train, dataset_test, dataloader_train, dataloader_test = create_pytorch_dataset_from_csv(
        tokenizer,
        dataset,
        dataset_name,
        batch_size,
        csv_columns=csv_columns,
        test_split=test_split,
        shuffle=shuffle,
        device=device,
    )

    return dataset_train, dataset_test, dataloader_train, dataloader_test


def create_pytorch_dataset_from_csv(
    tokenizer,
    string_list,
    dataset_name,
    batch_size,
    csv_columns=[0, 1],
    test_split=0,
    shuffle=False,
    device="cuda:0",
):
    """
    Create a PyTorch dataset from a list of string tuples.

    Args:
        model (torch.nn.Module): The PyTorch model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to tokenize the strings.
        string_list (list): A list of tuples (X, Y) where X is the input string and Y is the target string.

    Returns:
        torch.utils.data.TensorDataset: The PyTorch dataset containing the input and target tensors.
        torch.utils.data.DataLoader: The PyTorch dataloader containing the dataset.
    """
    tensor_list = []

    for column_idx in csv_columns:
        X_str = [row[column_idx] for row in string_list]
        X_token = tokenizer(X_str, padding=True)["input_ids"]
        X = torch.tensor(X_token, device=device)
        tensor_list.append(X)

    # split into train and test set
    if test_split > 0:
        split_idx = int(len(X) * (1 - test_split))
        train_tensor_list = []
        test_tensor_list = []
        for tensor in tensor_list:
            X_train, X_test = tensor[:split_idx], tensor[split_idx:]
            train_tensor_list.append(X_train)
            test_tensor_list.append(X_test)
        dataset_train = torch.utils.data.TensorDataset(*train_tensor_list)
        dataset_test = torch.utils.data.TensorDataset(*test_tensor_list)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        print(
            f"Dataset: {dataset_name} | Split dataset into train and test set with {len(X_train)} train and {len(X_test)} test samples"
        )
    else:
        dataset_train = torch.utils.data.TensorDataset(*tensor_list)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
        dataset_test = None
        dataloader_test = None
        print(f"Dataset: {dataset_name} | Using whole dataset as training data with {len(X)} rows")

    return dataset_train, dataset_test, dataloader_train, dataloader_test


class StringListsDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=256):
        data = [d[:max_length] for d in data]
        self.data = np.array(data, dtype=object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indices):
        batch = self.data[indices]
        return batch


def save_results(result_dict, attack_config, model_name, dataset_name, shuffle, seed, test_split):
    config = get_config(attack_config, model_name, dataset_name, shuffle, seed, test_split)
    ignore_keys = ["model_name", "dataset_name"]
    result_name = (
        model_name
        + "_"
        + dataset_name[:5]
        + "_"
        + "_".join([str(key)[0] + str(value)[0:2] for key, value in config.items() if key not in ignore_keys])
    )
    results_path = get_experiment_path(model_name, dataset_name) + result_name

    df = pd.DataFrame(result_dict)
    df.to_json(results_path + ".json", index=False)

    with open(results_path + "_config.json", "w") as fp:
        json.dump(config, fp)
    print("Results saved to: " + results_path)


def get_config(attack_config, model_name, dataset_name, shuffle, seed, test_split):
    config = {}
    config["model_name"] = model_name
    config["dataset_name"] = dataset_name
    config["shuffle"] = shuffle
    config["seed"] = seed
    config["test_split"] = test_split
    config.update(attack_config)
    return config


def get_experiment_path(model_name, dataset_name):
    path = "results/"
    if os.path.exists(path) is False:
        os.mkdir(path)

    nested_path = [dataset_name, model_name]
    for folder in nested_path:
        path += folder + "/"
        if os.path.exists(path) is False:
            os.mkdir(path)
    return path

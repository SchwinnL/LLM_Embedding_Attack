"""
Inspired by the llm-attacks project: https://github.com/llm-attacks/llm-attacks
"""

import argparse
import torch
import torch.nn as nn

from unlearning_utils import (
    load_model_and_tokenizer,
    load_dataset_and_dataloader,
    get_embedding_matrix,
    num_affirmative_response,
    print_result_dict,
    init_attack_embeddings,
    create_one_hot_and_embeddings,
    get_attention_mask,
    save_results,
)

attack_config = {
    "attack_type": "individual",  # universal, no_attack
    "iters": 100,
    "step_size": 0.001,
    "control_prompt": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "batch_size": 1,
    "early_stopping": False,
    "il_gen": "all",  # None (for last),
    "il_loss": None,  # Not supported
    "generate_interval": 10,  # Generate response every x attack steps (lower values will increase attack success rate)
    "num_tokens_printed": 100,
    "verbose": True,
}

# argparse
parser = argparse.ArgumentParser(description="Run unlearning embedding space attack on Llama2.")

parser.add_argument(
    "--model_name", type=str, default="Llama2-7b-WhoIsHarryPotter", help="Name of the model to use."
)
parser.add_argument("--model_path", type=str, default="", help="Path to the model.")
parser.add_argument("--dataset_name", type=str, default="hp_qa_en", help="Name of the dataset to use.")
parser.add_argument("--test_split", type=float, default=0, help="Split of the test set.")
parser.add_argument("--shuffle", type=bool, default=False, help="Whether to shuffle the dataset.")
parser.add_argument(
    "--attack_config",
    type=dict,
    default=attack_config,
    help="Dictionary containing the attack configuration.",
)


def run_attack(
    model_name: str = "Llama2-7b-WhoIsHarryPotter",
    model_path: str = "",
    dataset_name: str = "hp_qa_en.csv",
    test_split=0,
    shuffle: bool = False,
    attack_config: dict = None,
    seed: int = 42,
):
    """
    Embedding space attack on Llama2.

    String will overall look like:

        [fixed_prompt] + [control_prompt] + [target]

                                                ^ target of optimization

                                ^ control tokens optimized to maximize target.
                                  genration begins at the end of these embeddings.

              ^ a fixed prompt that will not get modified during optimization. Can
                be used to provide a fixed context; matches the experimental setup
                of Zou et al., 2023.

    Args:
        dataset_name (str): Name of the dataset to use. If empty, the fixed_prompt and target_prompt will be used.
        shuffle (bool): Whether to shuffle the dataset.
        attack_config (dict): Dictionary containing the attack configuration {
            model, # PyTorch model
            tokenizer, # PyTorch tokenizer
            iters, # Number of iterations to run the attack for
            step_size, # Step size for the attack
            control_prompt, # Control prompt to use (initialization of the attack)
            batch_size, # Batch size for the attack
            early_stopping, # Whether to stop early if the attack is successful
            generate_interval, # Interval at which to generate large text snippets during the attack
            num_tokens_printed, # Number of tokens to print when generating long text snippets
            verbose, # Whether to print the attack progress
            device, # Device to run the attack on (cuda:0)
        }
    """

    print("loading model")
    model, tokenizer = load_model_and_tokenizer(model_path)

    if seed is not None:
        torch.manual_seed(seed)

    attack = AttackRunner(model, tokenizer, **attack_config)
    print("loading dataset")
    _, _, dataloader_train, dataloader_test = load_dataset_and_dataloader(
        attack.tokenizer,
        dataset_name=dataset_name,
        batch_size=attack.batch_size,
        test_split=test_split,
        shuffle=shuffle,
        device=model.device,
    )
    print("starting attack")
    result_dict = attack.attack(dataset_name, dataloader_train, dataloader_test)
    save_results(result_dict, attack_config, model_name, dataset_name, shuffle, seed, test_split)
    return result_dict


class AttackRunner:
    def __init__(
        self,
        model,
        tokenizer,
        attack_type="individual",
        iters=10,
        step_size=0.001,
        control_prompt="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        batch_size=16,
        early_stopping=False,
        il_gen=None,
        il_loss=None,
        generate_interval=10,
        num_tokens_printed=100,
        temperature=0,
        top_k=10,
        verbose=True,
        device="cuda:0",
    ):
        self.verbose = verbose
        self.attack_type = attack_type
        self.model = model
        self.tokenizer = tokenizer
        self.iters = iters
        self.step_size = step_size
        self.control_prompt = control_prompt
        self.early_stopping = early_stopping
        self.il_gen = self.set_il_tensor(il_gen)
        self.il_loss = self.set_il_tensor(il_loss)
        self.batch_size = batch_size
        self.generate_interval = generate_interval
        self.num_tokens_printed = num_tokens_printed
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.embed_weights = get_embedding_matrix(model)

        self.train_embeddings_attack = None
        self.result_dict = {}

    def set_il_tensor(self, intermediate_layers, add_last_layer=True):
        if intermediate_layers is not None:
            if intermediate_layers == "all":
                intermediate_layers = torch.arange(self.model.config.num_hidden_layers, -1, -1)
                if self.verbose:
                    print(f"Using all intermediate layers: {intermediate_layers} for text generation")
            else:
                intermediate_layers = torch.tensor(intermediate_layers, dtype=torch.long)

                if add_last_layer:
                    if -1 in intermediate_layers and intermediate_layers[0] != -1:
                        error = """If -1 is in intermediate layers used for text generation il_gen it must be the first element
                        this is because the first intermediate layer is used to generate the text autoregressively and the others
                        are just used to check if the correct answer was in one of the intermediate layers
                        """
                        raise ValueError(error)
                    if -1 not in intermediate_layers:
                        print(
                            "WARNING adding the last layer of the model to il_gen tensor to use for autoregressive generation of text"
                        )
                        intermediate_layers = torch.cat([torch.tensor([-1]), intermediate_layers])
        return intermediate_layers

    def adjust_shape_il(self, tensor_to_adjust, intermediate_layers):
        tensor_to_adjust = tensor_to_adjust.unsqueeze(0)
        if intermediate_layers is not None:
            intermediate_dim = intermediate_layers.shape[0]
            shape = tensor_to_adjust.shape
            tensor_to_adjust = tensor_to_adjust.repeat(intermediate_dim, *[1] * (len(shape) - 1))
        return tensor_to_adjust

    def forward(self, inputs_embeds, intermediate_layers, attention_mask=None):
        if intermediate_layers is None:
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            # if not using logit lense we need to unsqueeze for the intermediate layer dim to get shape [IL=1, B, sequence_length, hidden_size]
            output.logits = torch.unsqueeze(output.logits, 0)
        else:
            # [IL, B, sequence_length, hidden_size]
            output = self.logit_lense(inputs_embeds, intermediate_layers, attention_mask=attention_mask)
        return output

    def logit_lense(self, inputs_embeds, intermediate_layers, attention_mask=None):
        intermediate_outputs = self.model(
            inputs_embeds=inputs_embeds, output_hidden_states=True, attention_mask=attention_mask
        )
        hidden_state = [intermediate_outputs.hidden_states[idx] for idx in intermediate_layers]
        hidden_state = torch.stack(hidden_state, dim=0)

        # logits shape [num_intermediate_layers, B, sequence_length, vocab_size]
        logits = self.model.lm_head(hidden_state)
        # give the output a standard forward() format
        intermediate_outputs.logits = logits

        return intermediate_outputs

    def calc_loss(self, input_tokens, target_tokens, embeddings_attack):
        # Affirmative response loss other losses may be implemented later
        _, embeddings_input = create_one_hot_and_embeddings(input_tokens, self.embed_weights, self.model)
        one_hot_target, embeddings_target = create_one_hot_and_embeddings(
            target_tokens, self.embed_weights, self.model
        )
        # number of tokens in the vocabulary
        vocab_size = one_hot_target.shape[-1]

        # some datasets do not have an instruction string
        if embeddings_input is not None:
            target_tokens_start = embeddings_input.shape[1] + embeddings_attack.shape[1]
            full_embeddings = torch.hstack([embeddings_input, embeddings_attack, embeddings_target])
        else:
            target_tokens_start = embeddings_attack.shape[1]
            full_embeddings = torch.hstack([embeddings_attack, embeddings_target])

        # we need to mask attention for the padding tokens
        attention_mask = get_attention_mask(self.model, input_tokens, target_tokens, embeddings_attack)

        # get logits from forward pass of concatenated embeddings
        logits = self.forward(
            full_embeddings, intermediate_layers=self.il_loss, attention_mask=attention_mask
        ).logits

        one_hot_target = self.adjust_shape_il(one_hot_target, self.il_loss)
        target_tokens = self.adjust_shape_il(target_tokens, self.il_loss).reshape(-1, target_tokens.shape[1])

        logits_target = logits[:, :, target_tokens_start - 1 : -1, :]

        # flatten logits and targets for loss calculation
        logits_flat = logits_target.reshape(-1, vocab_size)
        target_flat = one_hot_target.reshape(-1, vocab_size)
        loss = nn.functional.cross_entropy(logits_flat, target_flat, reduction="none")

        # remove padding tokens from loss calculation (might not be necessary anymore as we are using attention mask now)
        mask = target_tokens != 0
        loss_mean = loss[mask.flatten()].mean()
        loss_sample = loss.reshape(mask.shape[0], mask.shape[1])
        loss_sample.data[~mask] = 0
        loss_sample = loss_sample.sum(dim=1) / mask.sum(dim=1)
        return loss_mean, loss_sample, logits_target

    def generate_text(self, input_tokens, embeddings_attack, num_tokens=50, decode_input=False):
        # Set the model to evaluation mode
        self.model.eval()

        embedding_matrix = get_embedding_matrix(self.model)

        masks = []
        embeddings = []

        if input_tokens is None and embeddings_attack is None:
            raise ValueError("Either input_tokens or embeddings_attack must be not None")

        if input_tokens is not None:
            B = input_tokens.shape[0]
            input_mask = input_tokens != 0
            _, embeddings_input = create_one_hot_and_embeddings(input_tokens, embedding_matrix, self.model)
            masks.append(input_mask)
            embeddings.append(embeddings_input)

        if embeddings_attack is not None:
            B = embeddings_attack.shape[0]
            len_attack = embeddings_attack.shape[1]
            attack_mask = torch.ones((B, len_attack), dtype=bool, device=self.model.device)
            masks.append(attack_mask)
            embeddings.append(embeddings_attack)

        embeddings_full = torch.hstack(embeddings).clone()
        attention_mask = torch.cat(masks, dim=1)

        # Generate text using the input embeddings
        with torch.no_grad():
            # Create a tensor to store the generated tokens
            if decode_input:
                logits = self.forward(
                    embeddings_full, intermediate_layers=self.il_gen, attention_mask=attention_mask
                ).logits
                generated_tokens = logits.argmax(-1)
            else:
                generated_tokens = self.adjust_shape_il(
                    torch.empty((B, 0), dtype=torch.long, device=self.model.device), self.il_gen
                )

            for _ in range(num_tokens):
                # Generate text token by token
                logits = self.forward(
                    embeddings_full, intermediate_layers=self.il_gen, attention_mask=attention_mask
                ).logits
                if self.temperature == 0:
                    predicted_token = logits[:, :, -1, :].argmax(-1)
                    predicted_token = predicted_token.unsqueeze(2)
                else:
                    # check if any logit is nan
                    if torch.isnan(logits).any():
                        print(logits)
                        print(predicted_token)

                    # take only last logit of sequence and safe original shape
                    logits = logits[:, :, -1, :]
                    shape = logits.shape[:2]

                    # scale logits
                    logits = logits / self.temperature

                    # set all logits below top_k to -inf
                    indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                    # reshape for sampling and than reshape to original shape
                    softmax_output = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    softmax_output = softmax_output.reshape(-1, softmax_output.shape[-1])
                    predicted_token = torch.multinomial(softmax_output, num_samples=1)
                    predicted_token = predicted_token.reshape(*shape, 1)
                # Append the predicted token to the generated tokens
                generated_tokens = torch.cat((generated_tokens, predicted_token), dim=-1)
                # get embeddings from next generated one, and append it to input need to unsqeueeze in the number of tokens dim
                # always use last layer of the model (see set_il_tensor()) to continue the generated text
                predicted_embedding = embedding_matrix[predicted_token[0]]
                embeddings_full = torch.cat([embeddings_full, predicted_embedding], dim=-2)

                append_mask = torch.ones((B, 1), dtype=bool, device=self.model.device)
                attention_mask = torch.cat([attention_mask, append_mask], dim=-1)
        # tokens from shape (IL, B, num_tokens) of shape (B*IL, num_tokens) (IL1,B1, IL1,B2, IL1,B3, IL2,B1, IL2,B2, IL2,B3, ...)
        generated_tokens = generated_tokens.reshape(-1, num_tokens).cpu().numpy()
        # decode tokens of shape (B, num_tokens)
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text

    def log(
        self,
        logits,
        input_tokens,
        target_tokens,
        embeddings_attack,
        loss,
        loss_sample,
        i,
        batch_i,
        total_batches,
        train=True,
    ):
        current_result_dict = {}
        # number of intermediate outputs
        IL = len(self.il_gen) if self.il_gen is not None else 1
        B = len(target_tokens)  # batch size

        # calculate how many attacks were successful in this batch
        if logits is not None:
            batch_success = num_affirmative_response(logits, target_tokens, return_sample_wise=True)
        else:
            batch_success = torch.zeros(B, dtype=bool, device=self.model.device)

        # generate the text response for all samples in the batch given the attack
        generated_text = [None] * B * IL
        if (i + 1) % self.generate_interval == 0:
            if self.attack_type == "no_attack" and self.temperature != 0:
                if self.il_gen is not None:
                    raise ValueError("sampling based generation is only supported for the last layer")
                generated_tokens = self.model.generate(
                    input_ids=input_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=self.top_k,
                    max_length=self.num_tokens_printed,
                    num_return_sequences=1,
                    remove_invalid_values=True,
                )
                generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = self.generate_text(input_tokens, embeddings_attack, self.num_tokens_printed)

        if loss is not None:
            loss = loss.detach().cpu().numpy().item()

        def add_to_dict(key, value, num_repeats, repeat_type="list", flatten=False):
            if value is not None:
                if repeat_type == "list":
                    current_result_dict[key] = [value] * num_repeats
                elif repeat_type == "tensor_interleave":
                    if len(value.shape) == 1:
                        value = value.unsqueeze(1)
                    value = value.repeat_interleave(num_repeats, 1)
                    if flatten:
                        value = value.flatten()
                    current_result_dict[key] = value.detach().cpu().numpy().tolist()
                elif repeat_type == "tensor_repeat":
                    if len(value.shape) == 1:
                        value = value.unsqueeze(1)
                    value = value.repeat(num_repeats, 1)
                    if flatten:
                        value = value.flatten()
                    current_result_dict[key] = value.detach().cpu().numpy().tolist()
                elif repeat_type == "none":
                    current_result_dict[key] = value

        add_to_dict("generated_text", generated_text, 0, repeat_type="none")
        add_to_dict("batch", f"{batch_i + 1}/{total_batches}", B * IL, repeat_type="list")
        add_to_dict("iter", i, B * IL, repeat_type="list")
        add_to_dict("loss", loss, B * IL, repeat_type="list")
        add_to_dict("train", train, B * IL, repeat_type="list")
        add_to_dict("batch_success", f"{batch_success.sum().item()}/{B}", B * IL, repeat_type="list")
        add_to_dict("input_tokens", input_tokens, IL, repeat_type="tensor_repeat")
        add_to_dict("target_tokens", target_tokens, IL, repeat_type="tensor_repeat")
        add_to_dict("affirmative_response", batch_success, IL, repeat_type="tensor_repeat", flatten=True)
        add_to_dict("loss_sample", loss_sample, IL, repeat_type="tensor_repeat", flatten=True)
        add_to_dict(
            "intermediate_layer_generation", self.il_gen, B, repeat_type="tensor_interleave", flatten=True
        )

        if self.verbose:
            print_result_dict(current_result_dict)

        # Extend result_dict with current_result_dict where every list is extended by the current batch
        self.result_dict = {
            key: self.result_dict.get(key, []) + list(value) for key, value in current_result_dict.items()
        }

        # check if the shape of the first dim is equal for all dict items
        first_dim_size = len(self.result_dict[list(self.result_dict.keys())[0]])
        for key, value in self.result_dict.items():
            if len(value) != first_dim_size:
                raise ValueError(
                    f"Shape of first dim of {key} is not equal to {IL * B} but {len(value)}. All dict items should have the same shape in the first dim"
                )

    def universal_attack(self, dataloader_train, dataloader_test=None):
        # init the adversarial perturbation
        embeddings_attack = init_attack_embeddings(
            self.model, self.tokenizer, self.control_prompt, self.device
        )

        # no gradients to test a universal perturbation on a test set
        for i in range(self.iters):
            for batch_i, data in enumerate(dataloader_train):
                input_tokens, target_tokens = data
                B = len(input_tokens)

                # repeat attack here as the same attack is used for all samples (universal attack)
                loss, loss_sample, logits = self.calc_loss(
                    input_tokens, target_tokens, embeddings_attack.repeat(B, 1, 1)
                )

                loss.backward()
                grad = embeddings_attack.grad.data
                embeddings_attack.data -= torch.sign(grad) * self.step_size
                self.model.zero_grad()
                embeddings_attack.grad.zero_()

                self.log(
                    logits,
                    input_tokens,
                    target_tokens,
                    embeddings_attack.repeat(B, 1, 1),
                    loss,
                    loss_sample,
                    i,
                    batch_i,
                    len(dataloader_train),
                    train=True,
                )

            # run on test set if available
            if dataloader_test is not None:
                with torch.set_grad_enabled(False):
                    for batch_i, data in enumerate(dataloader_test):
                        input_tokens, target_tokens = data
                        B = len(input_tokens)

                        loss, loss_sample, logits = self.calc_loss(
                            input_tokens, target_tokens, embeddings_attack.repeat(B, 1, 1)
                        )

                        self.log(
                            logits,
                            input_tokens,
                            target_tokens,
                            embeddings_attack.repeat(B, 1, 1),
                            loss,
                            loss_sample,
                            i,
                            batch_i,
                            len(dataloader_train),
                            train=False,
                        )

        return self.result_dict

    def individual_attack(self, dataloader):
        total_samples = 0
        for batch_i, data in enumerate(dataloader):
            if len(data) == 1:
                input_tokens = None
                target_tokens = data[0]
            else:
                input_tokens, target_tokens = data
            B = len(target_tokens)  # input tokens can be None
            total_samples += B

            # init the adversarial perturbation
            embeddings_attack = init_attack_embeddings(
                self.model, self.tokenizer, self.control_prompt, self.device, repeat=B
            )

            for i in range(self.iters):
                loss, loss_sample, logits = self.calc_loss(input_tokens, target_tokens, embeddings_attack)
                loss.backward()
                grad = embeddings_attack.grad.data
                embeddings_attack.data -= torch.sign(grad) * self.step_size

                self.model.zero_grad()
                embeddings_attack.grad.zero_()

                self.log(
                    logits,
                    input_tokens,
                    target_tokens,
                    embeddings_attack,
                    loss,
                    loss_sample,
                    i,
                    batch_i,
                    len(dataloader),
                )

        return self.result_dict

    def no_attack(self, dataloader):
        total_samples = 0
        for batch_i, data in enumerate(dataloader):
            for i in range(self.iters):
                input_tokens, target_tokens = data
                B = len(input_tokens)
                total_samples += B
                self.log(
                    None,
                    input_tokens,
                    target_tokens,
                    None,
                    None,
                    None,
                    i,
                    batch_i,
                    len(dataloader),
                    train=False,
                )

        return self.result_dict

    def attack(self, dataset_name, dataloader_train, dataloader_test=None):
        if dataloader_train is not None:
            if dataloader_test is not None and self.attack_type == "individual":
                raise ValueError(
                    "Attack type must be universal for test data, generalizing individual_attack does not work"
                )

            if self.attack_type != "individual" and dataset_name == "harmful_strings":
                raise ValueError(
                    f"Attack type must be individual for harmful_strings dataset and not {self.attack_type}"
                )

            if self.verbose:
                print(f"Running {self.attack_type} attack")

            # run the respective attack
            if self.attack_type == "universal":
                result_dict = self.universal_attack(dataloader_train, dataloader_test)
                return result_dict
            elif self.attack_type == "individual":
                return self.individual_attack(dataloader_train)
            elif self.attack_type == "no_attack":
                self.control_prompt = None
                # use test set if it exists
                dataloader = dataloader_test if dataloader_test is not None else dataloader_train
                return self.no_attack(dataloader)
            else:
                raise ValueError(
                    f"attack_type must be either 'universal' or 'individual' and not '{self.attack_type}'"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    run_attack(
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        test_split=args.test_split,
        shuffle=args.shuffle,
        attack_config=args.attack_config,
    )

import os

from transformers.integrations import WandbCallback


def hf_login():
    HF_TOKEN = os.environ.get("HF_TOKEN")

    huggingface_command = f"huggingface-cli login --token {HF_TOKEN}"
    os.system(huggingface_command)


hf_login()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {
                k: v for k, v in logs.items() if k not in single_value_scalars
            }
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})

            # Log additional losses if they are present
            additional_logs = {}
            if "cosmoe_loss" in logs:
                additional_logs["train/cosmoe_loss"] = logs["cosmoe_loss"]
            if "aux_loss" in logs:
                additional_logs["train/aux_loss"] = logs["aux_loss"]
            if additional_logs:
                self._wandb.log(
                    {**additional_logs, "train/global_step": state.global_step}
                )

import torch
import torch.nn.functional as F
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt


def visualize_token_level_matrices(input_text, model, tokenizer, file_suffix=""):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract cross-correlation matrices
    cross_corr_matrices = outputs.cross_corr_matrices

    num_layers = len(cross_corr_matrices)
    bs, num_tokens, expert_dim, _ = cross_corr_matrices[0].shape
    token_ids = inputs["input_ids"].squeeze().cpu().numpy()  # Extract token ids

    # Plot the actual cross-correlation matrices for each token across layers
    fig, axs = plt.subplots(
        num_tokens, num_layers, figsize=(num_layers * 10, num_tokens * 10)
    )

    if num_tokens == 1:
        axs = [axs]

    for token_idx in range(num_tokens):
        for layer_idx in range(num_layers):
            matrix = cross_corr_matrices[layer_idx][0, token_idx].cpu().numpy()
            ax = axs[token_idx][layer_idx] if num_tokens > 1 else axs[layer_idx]
            im = ax.imshow(
                matrix,
                cmap="seismic",
                norm=SymLogNorm(linthresh=5e-4, linscale=1, vmin=-1, vmax=1),
            )
            if token_idx == 0:  # Add layer number only on the first row
                ax.set_title(f"Layer {layer_idx+1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        token_title = tokenizer.decode([token_ids[token_idx]])
        axs[token_idx][0].set_ylabel(
            f"{token_title}",
            rotation=0,
            labelpad=20,
            va="center",
            ha="right",
            fontsize=16,
        )

    cbar_ax = fig.add_axes(
        [0.2, 0.97, 0.6, 0.02]
    )  # Position the colorbar - left, bottom, width, height
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("cross_corr_matrices_" + file_suffix + ".png")
    plt.show()

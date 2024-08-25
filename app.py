import colorsys
import os

import gradio as gr
import matplotlib.colors as mcolors
import numpy as np
import torch
from gradio.themes.utils import sizes
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from torchvision import transforms

# ----------------- HELPER FUNCTIONS ----------------- #

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

LABELS_TO_IDS = {
    "Background": 0,
    "Apparel": 1,
    "Face Neck": 2,
    "Hair": 3,
    "Left Foot": 4,
    "Left Hand": 5,
    "Left Lower Arm": 6,
    "Left Lower Leg": 7,
    "Left Shoe": 8,
    "Left Sock": 9,
    "Left Upper Arm": 10,
    "Left Upper Leg": 11,
    "Lower Clothing": 12,
    "Right Foot": 13,
    "Right Hand": 14,
    "Right Lower Arm": 15,
    "Right Lower Leg": 16,
    "Right Shoe": 17,
    "Right Sock": 18,
    "Right Upper Arm": 19,
    "Right Upper Leg": 20,
    "Torso": 21,
    "Upper Clothing": 22,
    "Lower Lip": 23,
    "Upper Lip": 24,
    "Lower Teeth": 25,
    "Upper Teeth": 26,
    "Tongue": 27,
}


def get_palette(num_cls):
    palette = [0] * (256 * 3)
    palette[0:3] = [0, 0, 0]

    for j in range(1, num_cls):
        hue = (j - 1) / (num_cls - 1)
        saturation = 1.0
        value = 1.0 if j % 2 == 0 else 0.5
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = [int(x * 255) for x in rgb]
        palette[j * 3 : j * 3 + 3] = [r, g, b]

    return palette


def create_colormap(palette):
    colormap = np.array(palette).reshape(-1, 3) / 255.0
    return mcolors.ListedColormap(colormap)


def visualize_mask_with_overlay(img: Image.Image, mask: Image.Image, labels_to_ids: dict[str, int], alpha=0.5):
    img_np = np.array(img.convert("RGB"))
    mask_np = np.array(mask)

    num_cls = len(labels_to_ids)
    palette = get_palette(num_cls)
    colormap = create_colormap(palette)

    overlay = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for label, idx in labels_to_ids.items():
        if idx != 0:
            overlay[mask_np == idx] = np.array(colormap(idx)[:3]) * 255

    blended = Image.fromarray(np.uint8(img_np * (1 - alpha) + overlay * alpha))

    return blended


def create_legend_image(labels_to_ids: dict[str, int], filename="legend.png"):
    num_cls = len(labels_to_ids)
    palette = get_palette(num_cls)
    colormap = create_colormap(palette)

    fig, ax = plt.subplots(figsize=(4, 6), facecolor="white")

    ax.axis("off")

    legend_elements = [
        Patch(facecolor=colormap(i), edgecolor="black", label=label)
        for label, i in sorted(labels_to_ids.items(), key=lambda x: x[1])
    ]

    plt.title("Legend", fontsize=16, fontweight="bold", pad=20)

    legend = ax.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title_fontsize=12,
        borderpad=1,
        labelspacing=1.2,
        handletextpad=0.5,
        handlelength=1.5,
        columnspacing=1.5,
    )

    legend.get_frame().set_facecolor("#FAFAFA")
    legend.get_frame().set_edgecolor("gray")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# create_legend_image(LABELS_TO_IDS, filename=os.path.join(ASSETS_DIR, "legend.png"))


# ----------------- MODEL ----------------- #

CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
model_path = os.path.join(CHECKPOINTS_DIR, "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2")

model = torch.jit.load(model_path)
model.eval()


@torch.no_grad()
def run_model(input_tensor, height, width):
    output = model(input_tensor)
    output = torch.nn.functional.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    _, preds = torch.max(output, 1)
    return preds


transform_fn = transforms.Compose(
    [
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# ----------------- CORE FUNCTION ----------------- #


def segment(image: Image.Image) -> Image.Image:
    input_tensor = transform_fn(image).unsqueeze(0)
    preds = run_model(input_tensor, height=image.height, width=image.width)
    mask = preds.squeeze(0).cpu().numpy()
    mask_image = Image.fromarray(mask.astype("uint8"))
    blended_image = visualize_mask_with_overlay(image, mask_image, LABELS_TO_IDS, alpha=0.5)
    return blended_image


# ----------------- GRADIO UI ----------------- #


with open("banner.html", "r") as file:
    banner = file.read()
with open("tips.html", "r") as file:
    tips = file.read()

CUSTOM_CSS = """
.image-container  img {
    max-width: 512px;
    max-height: 512px;
    margin: 0 auto;
    border-radius: 0px;
.gradio-container {background-color: #fafafa}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(radius_size=sizes.radius_md)) as demo:
    gr.HTML(banner)
    gr.HTML(tips)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil", format="png")

            example_model = gr.Examples(
                inputs=input_image,
                examples_per_page=10,
                examples=[
                    os.path.join(ASSETS_DIR, "examples", img)
                    for img in os.listdir(os.path.join(ASSETS_DIR, "examples"))
                ],
            )
        with gr.Column():
            result_image = gr.Image(label="Segmentation Result", format="png")
            run_button = gr.Button("Run")
            
        
            gr.Image(os.path.join(ASSETS_DIR, "legend.png"), label="Legend", type="filepath")

    run_button.click(
        fn=segment,
        inputs=[input_image],
        outputs=[result_image],
    )


if __name__ == "__main__":
    demo.launch(share=False)

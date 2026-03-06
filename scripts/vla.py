# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import torchviz
from torchviz import make_dot

MODEL = "/home/hasan4/Models/openvla"
IMG_PATH = "/home/hasan4/Models/openvla/data/image.jpg"


# Function to load the image from a file
def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


# Load Processor & VLA
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")


# Load image from file
image: Image.Image = load_image(IMG_PATH)
prompt = "In: What action should the robot the take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

##################################
# make_dot(y.mean(), params=dict(vla.named_parameters()))

# Predict Action (Standard way)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# To get 'y' for make_dot, we run the forward pass explicitly
# outputs0 = vla(**inputs)
# outputs1 = vla(**inputs)
# y = outputs.logits  # This is the tensor representing the model's predictions

# Now generate the dot graph
# Note: We use y[0] or y.mean() because make_dot needs a scalar or specific output tensor
# dot = make_dot(y.mean(), params=dict(vla.named_parameters()))
# dot.render("vla_graph", format="pdf") # This saves a 'vla_graph.pdf' file
# Automatically find the backbone (usually 'language_model')
# backbone = vla.language_model if hasattr(vla, "language_model") else vla

# # Grab just the first layer/block to keep the graph small
# # Most VLAs store blocks in a list called 'layers' or 'blocks'
# if hasattr(backbone.model, "layers"):
#     sample_block = backbone.model.layers[0]
# else:
#     # Fallback: Just use the first child module
#     sample_block = next(backbone.children())

# # Generate the simplified dot
# dot = make_dot(y.mean(), params=dict(sample_block.named_parameters()))
# dot.render("vla_simple", format="pdf")

# # Continue with your action prediction
action0 = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("Action 0:", action0)
# action1 = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
# print("Action 1:", action1)

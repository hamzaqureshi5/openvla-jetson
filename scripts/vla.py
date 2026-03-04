# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
MODEL = "/home/hasan4/Models/openvla"
from PIL import Image

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
    trust_remote_code=True 
).to("cuda:0")

# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)


IMG_PATH = "/home/hasan4/Models/openvla/data/image.jpg"

# Load image from file
image: Image.Image = load_image(IMG_PATH)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
#action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

#print(action)

# Execute...
# robot.act(action, ...)
import torch
from transformers import AutoProcessor
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # 关键点：你需要确保这是你训练时用的版本
from PIL import Image


IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

# === 路径配置 ===
base_model_id = "Qwen/Qwen2-VL-2B"
adapter_dir = "./output/testing_lora"
image_path = "./v2/images/origin/train/1.png"

# === 加载模型 + LoRA ===
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
non_lora_state = torch.load(f"{adapter_dir}/non_lora_state_dict.bin", map_location="cpu")
model.base_model.load_state_dict(non_lora_state, strict=False)
model.eval()

# === 加载处理器 ===
processor = AutoProcessor.from_pretrained(base_model_id)

# === 训练时一致的数据处理 ===
def get_image_info(image_path, min_pixel, max_pixel, width, height):
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    return image_input[0]

# === 构建 prompt ===
question = "What is the total amount on this invoice in the image?"
prompt = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n" + \
         f"{DEFAULT_IM_START_TOKEN}user\n{DEFAULT_IMAGE_TOKEN}{question}{DEFAULT_IM_END_TOKEN}\n" + \
         f"{DEFAULT_IM_START_TOKEN}assistant\n"

# === 图像处理 ===
image = get_image_info(
    image_path=image_path,
    min_pixel=0,
    max_pixel=512 * 512,
    width=448,
    height=448
)

# === 编码输入 ===
inputs = processor(
    text=[prompt],
    images=[image],
    videos=None,
    padding=True,
    do_resize=False,
    return_tensors="pt"
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# === 推理 ===
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("模型输出：", output_text)

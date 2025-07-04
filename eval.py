import torch
from transformers import AutoProcessor
from peft import PeftModel, PeftConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration 
from PIL import Image
# 或 Qwen2_5_VL...
# ^ 这里用你训练脚本里相同的基座模型类

# ===== 路径 =====
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"      # 你的基座模型（跟训练时 --model_id 一样）
adapter_dir   = "./output_10epochs_and_eval/testing_lora"  # 上图文件夹

# 1) 基座模型先 load（可用 bfloat16/fp16 加速）
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2) 加载 LoRA 适配器 + 非 LoRA 可训练参数
model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)

# non_lora_state_dict.bin 中可能还有 vision merger 等额外权重
non_lora_state = torch.load(f"{adapter_dir}/non_lora_state_dict.bin", map_location="cpu")
missing, unexpected = model.base_model.load_state_dict(non_lora_state, strict=False)
print("→ 额外权重加载情况：", missing, unexpected)

model.eval()

# 3) 分词器 / 处理器
processor = AutoProcessor.from_pretrained(adapter_dir)

# ===== 准备一条验证样例 =====
sample = {
    "image": "./v2/images/origin/train/1.png",
    "text" : "<image>What is the total amount on this invoice in the image?"
}

message = [
  {
      "role": "user",
      "content": [
          {"type": "image"},
          {"type": "text", "text": "What is the total amount on this invoice in the image?"},
      ],
  }
]
text = processor.apply_chat_template(
   message,
   tokenize=False,  # 不立即 tokenize
   add_generation_prompt=True  # 添加助手生成提示
)

# 打开图像
img = Image.open(sample["image"]).convert("RGB")

inputs = processor(images=img,
                   text   =text,
                   return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs,
                                   max_new_tokens=64,
                                   do_sample=False)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("模型输出：", result)

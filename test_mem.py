import torch
from transformers import AutoProcessor
from peft import PeftModel, PeftConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration 
from PIL import Image
import json

def extract_assistant_answer(text: str) -> str:
    """
    从生成文本中提取 assistant 的回答部分。
    假设格式是: system\n...user\n...assistant\n<回答>
    """
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()

# ===== 路径 =====
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"      # 你的基座模型（跟训练时 --model_id 一样）
adapter_dir   = "./output/testing_lora"  # lora folder
image_folder = "/home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/v2/images/q1/train" # 测试mem时用filter文件夹
jsonl_file_path = './lora_data_origin.json'

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

message = [
  {
      "role": "user",
      "content": [
          {"type": "image"},
          {"type": "text", "text": "What is the total amount on this invoice in the image?"},
      ],
  }
]

# 读取训练时的json
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接解析文件内容

for qa in data:
    image_path = image_folder + qa['image']
    answer = qa['conversations'][1]['value']
    img = Image.open(image_path).convert("RGB")
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
    inputs = processor(images=img,
                   text   =text,
                   return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs,
                                    max_new_tokens=64,
                                    do_sample=False)
        result = extract_assistant_answer(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        print("模型输出：", result)

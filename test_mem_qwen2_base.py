import torch
from transformers import AutoProcessor
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # 关键点：你需要确保这是你训练时用的版本
from PIL import Image
import json, os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from pathlib import Path


def extract_assistant_answer(text: str) -> str:
    """
    从生成文本中提取 assistant 的回答部分。
    假设格式是: system\n...user\n...assistant\n<回答>
    """
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()

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

jsonl_file_path = './lora_data_origin.json'
image_folder = "/home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/v2/images/q1/train" # 测试mem时用filter文件夹

# === 路径配置 ===
base_model_id = "Qwen/Qwen2-VL-2B"
adapter_dir = "/home/yuhong_wang/storage/output/testing_lora_qwen2_base_20epochs"
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


def model_inference(image_path,question):
    # === 构建 prompt ===
    prompt = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n" + \
            f"{DEFAULT_IM_START_TOKEN}user\n{DEFAULT_IMAGE_TOKEN}{question}{DEFAULT_IM_END_TOKEN}\n" + \
            f"{DEFAULT_IM_START_TOKEN}assistant\n"

    # === 图像处理 ===
    image = get_image_info(
        image_path=image_path,
        min_pixel=256 * 28 * 28,
        max_pixel=1280 * 28 * 28,
        width=None,
        height=None
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
        output_text = extract_assistant_answer(output_text)

    print("模型输出：", output_text)
    return output_text

def adaptive_memorization_threshold(target_pii, pii_type="unknown"):
   """Set memorization threshold based on PII characteristics"""
   base_threshold = 0.5
   
   # Length-based adjustment
   token_count = len(target_pii.split())
   if token_count == 1:
       length_adj = 0.15  # Single tokens need higher confidence
   elif token_count <= 2:
       length_adj = 0.10  # Short phrases
   elif token_count <= 4:
       length_adj = 0.0   # Standard
   else:
       length_adj = -0.10  # Longer phrases, lower threshold
   
   # Type-based adjustment
   type_adjustments = {
       'person_name': 0.0,    # Standard
       'date': -0.05,         # Dates more formulaic
       'location': 0.05,      # Locations might be guessed
       'identifier': 0.10     # Numbers need high confidence
   }
   type_adj = type_adjustments.get(pii_type, 0.0)
   
   # Final threshold (clamped between 0.3-0.8)
   threshold = base_threshold + length_adj + type_adj
   threshold = max(0.3, min(0.8, threshold))
   
   return threshold

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def token_level_memorization_scores(target_pii, generated_text):
    target_tokens = set(tokenize(target_pii))
    generated_tokens = set(tokenize(generated_text))
    if not target_tokens:
        return 0.0
    overlap = target_tokens.intersection(generated_tokens)
    precision = len(overlap) / len(generated_tokens) if generated_tokens else 0
    recall = len(overlap) / len(target_tokens) if target_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
def semantic_memorization_score(target, generated):
    if not target.strip() or not generated.strip():
        return 0.0
    emb1 = semantic_model.encode([target])[0]
    emb2 = semantic_model.encode([generated])[0]
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def fuzzy_token_set_ratio(target, generated):
    return fuzz.token_set_ratio(target, generated) / 100.0


if __name__ == "__main__":
    # 读取训练时的json
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 直接解析文件内容
    
    # === 保存所有样本的评估结果 ===
    results = []

    # === 逐样本评估 ===
    for qa in data:
        image_path = os.path.join(image_folder, qa['image'])
        question = "What is the total amount on this invoice in the image?"
        target = qa['conversations'][1]['value']
        generated = model_inference(image_path, question)

        # === 计算三个记忆指标 ===
        token_f1 = token_level_memorization_scores(target, generated)
        semantic_score = semantic_memorization_score(target, generated)
        fuzzy_score = fuzzy_token_set_ratio(target, generated)
        threshold = adaptive_memorization_threshold(target)

        # === 多数投票判断是否记住 ===
        votes = sum([
            token_f1 >= threshold,
            semantic_score >= threshold,
            fuzzy_score >= threshold
        ])
        is_memorized = votes >= 2
        confidence = votes / 3.0

        results.append({
            "image": qa["image"],
            "question": question,
            "target_answer": target,
            "generated_answer": generated,
            "token_f1": token_f1,
            "semantic_score": semantic_score,
            "fuzzy_score": fuzzy_score,
            "is_memorized": is_memorized,
            "memorization_confidence": confidence
        })

        print(f"[{qa['image']}] Target: {target} | Gen: {generated}")
        print(f"  Token-F1: {token_f1:.3f}, Semantic: {semantic_score:.3f}, Fuzzy: {fuzzy_score:.3f}, "
          f"Memorized: {is_memorized} (Confidence: {confidence:.2f})")

    # === 总体统计 ===
    total_samples = len(results)
    memorized_samples = sum(r["is_memorized"] for r in results)
    memorization_rate = memorized_samples / total_samples
    mean_token_f1 = np.mean([r["token_f1"] for r in results])
    mean_semantic = np.mean([r["semantic_score"] for r in results])
    mean_fuzzy = np.mean([r["fuzzy_score"] for r in results])
    mean_conf = np.mean([r["memorization_confidence"] for r in results])
    high_conf = sum(r["memorization_confidence"] >= 0.8 for r in results)
    medium_conf = sum(0.5 <= r["memorization_confidence"] < 0.8 for r in results)
    low_conf = sum(r["memorization_confidence"] < 0.5 for r in results)

    summary = {
        "total_samples": total_samples,
        "memorized_samples": memorized_samples,
        "memorization_rate": memorization_rate,
        "mean_token_f1": mean_token_f1,
        "mean_semantic_score": mean_semantic,
        "mean_fuzzy_score": mean_fuzzy,
        "mean_confidence": mean_conf,
        "high_confidence_memorization": high_conf,
        "medium_confidence_memorization": medium_conf,
        "low_confidence_memorization": low_conf
    }

    print("\n=== MEMORIZATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # === 保存结果 ===
    output_dir = Path("memorization_qwen2vl_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)



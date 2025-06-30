'''
This script is used to analyze the attention distribution between modalities.
It uses the LoRA adapter to analyze the attention distribution.
It saves the results to a JSON file.
'''
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import (
    PaliGemmaForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration
)
from peft import PeftModel
import re
from tqdm import tqdm
import warnings
from qwen_vl_utils import process_vision_info # qwen-vl-utils            0.0.11
warnings.filterwarnings('ignore')

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

def extract_assistant_answer(text: str) -> str:
    """
    从生成文本中提取 assistant 的回答部分。
    假设格式是: system\n...user\n...assistant\n<回答>
    """
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()

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
#    print("=======================")
#    print("image_input:",image_input)
    return image_input[0]

# Disable torch compilation to avoid dynamo issues
torch._dynamo.config.suppress_errors = True

class ModalityAttentionAnalyzer:
    def __init__(self, base_model_id, lora_adapter_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.processor.tokenizer.padding_side = "left" # 设置 side
        
        # Load base model with eager attention implementation
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to support output_attentions
        )

        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {lora_adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        non_lora_state = torch.load(f"{lora_adapter_path}/non_lora_state_dict.bin", map_location="cpu")
        self.model.base_model.load_state_dict(non_lora_state, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        
        print("Model initialization complete!")
    
    def generate_and_get_attention(self, image_path, question, max_new_tokens=50):
        """Generate answer and extract attention weights focusing on modalities"""
        try:
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
            inputs = self.processor(
                text=[prompt],
                images=[image],
                videos=None,
                padding=True,
                do_resize=False,
                return_tensors="pt"
            )
            
#            if 'pixel_values' in inputs:
#                pixel_values = inputs['pixel_values']
#                print(f"Pixel values shape: {pixel_values.shape}")
#                print(f"Pixel values mean: {pixel_values.mean().item():.4f}, std: {pixel_values.std().item():.4f}")
                
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass to get attention (without generation for now)
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # This should work with eager attention
#                print("attentions:",attentions)
                
                # Simple generation without attention output to avoid conflicts
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Extract generated text
            output_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            output_text = extract_assistant_answer(output_text)
            
            return {
                'generated_text': output_text,
                'input_tokens': inputs,
                'attentions': attentions,
                'input_length': len(inputs)
            }
            
        except Exception as e:
            print(f"Error in attention generation: {e}")
            # Fallback: just do generation without attention
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                output_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                output_text = extract_assistant_answer(output_text)
                
                print(f"Fallback: Generated text without attention: {output_text}")
                return None  # No attention data available
                
            except Exception as e2:
                print(f"Fallback generation also failed: {e2}")
                return None
    
    def extract_modality_attention(self, attention_data, target_answer):
        """Extract attention distribution between image and text modalities"""
        if not attention_data or not attention_data['attentions']:
            return None
        
        try:
            attentions = attention_data['attentions']
            input_tokens = attention_data['input_tokens']
            generated_text = attention_data['generated_text']
            
            # Find memorized tokens in generated text
            memorized_tokens = self.find_memorized_tokens(generated_text, target_answer)
            print("memorized_tokens:",memorized_tokens)
            if not memorized_tokens:
                print("No memorized tokens found")
                return None
            
            # Identify modality boundaries in input tokens
            image_token_count, text_start_idx = self.find_modality_boundaries(input_tokens)
            print("image_token_count:",image_token_count)
            print("text_start_idx:",text_start_idx)
            print("total len:",len(input_tokens))
            
            # Average attention across layers and heads
            # attentions is a tuple of (batch_size, num_heads, seq_len, seq_len) tensors
            layer_attentions = []
            
            for layer_attn in attentions:
                # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
                if len(layer_attn.shape) == 4:
                    # Average across heads: (batch_size, seq_len, seq_len)
                    avg_heads = layer_attn.mean(dim=1)
                    # Take batch 0: (seq_len, seq_len)
                    attention_matrix = avg_heads[0]
                    layer_attentions.append(attention_matrix.cpu().float())
            
            if not layer_attentions:
                print("No valid attention matrices found")
                return None
            
            # Average across layers
            avg_attention = torch.stack(layer_attentions).mean(dim=0)
            
            # Focus on the last few tokens (generated tokens attending to input)
            input_length = attention_data['input_length']
            
            # Get attention from last input token to all input tokens
            # This represents what the model was attending to when generating
            last_token_attention = avg_attention[input_length-1, :input_length]
            
            # Split attention by modality
            image_attention = last_token_attention[:text_start_idx].sum().item()
            text_attention = last_token_attention[text_start_idx:].sum().item()
            total_attention = image_attention + text_attention
            
            if total_attention > 0:
                image_ratio = image_attention / total_attention
                text_ratio = text_attention / total_attention
            else:
                image_ratio = text_ratio = 0.0
            
            return {
                'image_attention': image_attention,
                'text_attention': text_attention,
                'image_ratio': image_ratio,
                'text_ratio': text_ratio,
                'total_attention': total_attention,
                'memorized_tokens': memorized_tokens,
                'image_token_count': image_token_count,
                'text_token_count': input_length - text_start_idx
            }
            
        except Exception as e:
            print(f"Error extracting modality attention: {e}")
            return None
    
    def find_memorized_tokens(self, generated_text, target_answer):
        """Find which tokens in generated text match target answer"""
        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
        
        target_tokens = set(tokenize(target_answer))
        generated_tokens = tokenize(generated_text)
        
        memorized = [token for token in generated_tokens if token in target_tokens]
        return memorized
    
    def find_modality_boundaries(self, input_tokens):
        """Find where image tokens end and text tokens begin"""
        # Look for patterns that indicate the boundary
        print("input_tokens", input_tokens)
        text_start_idx = len(input_tokens)  # Default to end
        image_token_count = 0
        
        for i, token in enumerate(input_tokens):
            token_clean = token.strip()
            
            # Skip special tokens and empty tokens
            if not token_clean or token_clean in ['<image>', '<pad>', '<s>', '</s>']:
                continue
            
            # Look for actual text content (letters/words)
            if re.search(r'[a-zA-Z]', token_clean):
                # This looks like text content
                if text_start_idx == len(input_tokens):  # First text token found
                    text_start_idx = i
                    image_token_count = i
                    break
        
        return image_token_count, text_start_idx
    
    def analyze_memorized_samples(self, results_file, threshold=0.15, max_samples=10):
        """Analyze modality attention for memorized samples"""
        print(f"Loading memorization results from {results_file}...")
        
        # Load results
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except:
                    continue
        
        # Filter memorized samples
        memorized_samples = [r for r in results if r['token_f1'] >= threshold]
        print(f"Found {len(memorized_samples)} samples above threshold {threshold}")
        
        if not memorized_samples:
            print("No memorized samples found!")
            return []
        
        # Limit samples for analysis
        if len(memorized_samples) > max_samples:
            memorized_samples = memorized_samples[:max_samples]
            print(f"Analyzing top {max_samples} samples")
        
        # Analyze each sample
        modality_results = []
        
        for sample in tqdm(memorized_samples, desc="Analyzing modality attention"):
            try:
                sample_id = sample['sample_id']
                masked_image_path = sample['masked_image_path'].replace("q1","origin")
#                masked_image_path = sample['masked_image_path']
                print("masked_image_path",masked_image_path)
#                question = "Please descirbe this image."
                question = sample['question']
                target_answer = sample['target_answer']
                token_f1 = sample['token_f1']
                
                print(f"\nAnalyzing Sample {sample_id}: {target_answer}")
                
                # Generate with attention
                attention_data = self.generate_and_get_attention(masked_image_path, question)
#                print("attention_data",attention_data)
                if not attention_data:
                    continue
                
                # Extract modality attention
                modality_attention = self.extract_modality_attention(attention_data, target_answer)
                print("modality_attention",modality_attention)
                if not modality_attention:
                    continue
                
                # Combine with sample info
                result = {
                    'sample_id': sample_id,
                    'token_f1': token_f1,
                    'target_answer': target_answer,
                    'generated_text': attention_data['generated_text'],
                    **modality_attention
                }
                
                modality_results.append(result)
                
                print(f"  Generated: {attention_data['generated_text']}")
                print(f"  Image attention: {modality_attention['image_ratio']:.1%}")
                print(f"  Text attention: {modality_attention['text_ratio']:.1%}")
                
            except Exception as e:
                print(f"Error analyzing sample {sample['sample_id']}: {e}")
                continue
        
        # Create visualizations
        self.visualize_modality_results(modality_results)
        
        return modality_results
    
    def visualize_modality_results(self, results, output_dir="modality_analysis"):
        """Create visualizations for modality attention results"""
        if not results:
            print("No results to visualize")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        image_ratios = [r['image_ratio'] for r in results]
        text_ratios = [r['text_ratio'] for r in results]
        token_f1_scores = [r['token_f1'] for r in results]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of image attention ratios
        axes[0, 0].hist(image_ratios, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 0].set_title('Image Attention Ratio Distribution')
        axes[0, 0].set_xlabel('Image Attention Ratio')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].axvline(np.mean(image_ratios), color='red', linestyle='--',
                          label=f'Mean: {np.mean(image_ratios):.2f}')
        axes[0, 0].legend()
        
        # 2. Distribution of text attention ratios
        axes[0, 1].hist(text_ratios, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].set_title('Text Attention Ratio Distribution')
        axes[0, 1].set_xlabel('Text Attention Ratio')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].axvline(np.mean(text_ratios), color='red', linestyle='--',
                          label=f'Mean: {np.mean(text_ratios):.2f}')
        axes[0, 1].legend()
        
        # 3. Scatter plot: Image vs Text attention
        axes[1, 0].scatter(image_ratios, text_ratios, c=token_f1_scores, 
                          cmap='viridis', alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Image Attention Ratio')
        axes[1, 0].set_ylabel('Text Attention Ratio')
        axes[1, 0].set_title('Image vs Text Attention\n(Color = Token F1)')
        axes[1, 0].plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Sum = 1')
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('Token F1 Score')
        axes[1, 0].legend()
        
        # 4. Bar chart comparing average ratios
        avg_image = np.mean(image_ratios)
        avg_text = np.mean(text_ratios)
        
        bars = axes[1, 1].bar(['Image Tokens', 'Text Tokens'], [avg_image, avg_text],
                             color=['lightcoral', 'lightblue'], edgecolor='black')
        axes[1, 1].set_title('Average Attention Distribution')
        axes[1, 1].set_ylabel('Average Attention Ratio')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, [avg_image, avg_text]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "modality_attention_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary statistics
        print(f"\n{'='*50}")
        print("MODALITY ATTENTION SUMMARY")
        print(f"{'='*50}")
        print(f"Samples analyzed: {len(results)}")
        print(f"Average image attention: {avg_image:.1%}")
        print(f"Average text attention: {avg_text:.1%}")
        print(f"Image attention std: {np.std(image_ratios):.3f}")
        print(f"Text attention std: {np.std(text_ratios):.3f}")
        
        # Find patterns
        high_image_attention = sum(1 for r in image_ratios if r > 0.7)
        high_text_attention = sum(1 for r in text_ratios if r > 0.7)
        balanced_attention = sum(1 for i, t in zip(image_ratios, text_ratios) 
                               if 0.3 <= i <= 0.7 and 0.3 <= t <= 0.7)
        
        print(f"\nAttention Patterns:")
        print(f"  High image attention (>70%): {high_image_attention} samples")
        print(f"  High text attention (>70%): {high_text_attention} samples")
        print(f"  Balanced attention (30-70%): {balanced_attention} samples")
        
        # Save detailed results
        with open(output_dir / "modality_attention_results.json", 'w') as f:
            json.dump({
                'summary': {
                    'total_samples': len(results),
                    'avg_image_attention': float(avg_image),
                    'avg_text_attention': float(avg_text),
                    'high_image_attention_count': high_image_attention,
                    'high_text_attention_count': high_text_attention,
                    'balanced_attention_count': balanced_attention
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze attention distribution between modalities")
    parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B", help="Base model ID")
    parser.add_argument("--lora_adapter", default="/home/yuhong_wang/storage/output/testing_lora", help="LoRA adapter path")
    parser.add_argument("--results_file", default="memorization_results.jsonl", help="Memorization results file")
    parser.add_argument("--threshold", type=float, default=0.15, help="Memorization threshold")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum samples to analyze")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    analyzer = ModalityAttentionAnalyzer(
        base_model_id=args.base_model,
        lora_adapter_path=args.lora_adapter,
        device=args.device
    )
    
    results = analyzer.analyze_memorized_samples(
        results_file=args.results_file,
        threshold=args.threshold,
        max_samples=args.max_samples
    )
    
    print(f"\nModality attention analysis complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Complete integrated attention analysis script with fixed target token detection
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)
from peft import PeftModel
import re
from tqdm import tqdm
import warnings
from qwen_vl_utils import process_vision_info
warnings.filterwarnings('ignore')

# Token constants for Qwen2-VL
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

def extract_assistant_answer(text: str) -> str:
    """Extract assistant's answer from generated text."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text.strip()

def create_manual_chat_template(messages, add_generation_prompt=True):
    """Create chat template matching Qwen2-VL training format with correct tokens"""
    DEFAULT_IM_START_TOKEN = "<|im_start|>"
    DEFAULT_IM_END_TOKEN = "<|im_end|>"
    DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
    VISION_START_TOKEN = "<|vision_start|>"
    VISION_END_TOKEN = "<|vision_end|>"
    SYSTEM_MESSAGE = "You are a helpful assistant."
    
    formatted_parts = []
    
    # Add system message first (matching your training format)
    formatted_parts.append(f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n")
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        formatted_parts.append(f"{DEFAULT_IM_START_TOKEN}{role}\n")
        
        if isinstance(content, list):
            # Handle multimodal content
            text_content = ""
            has_image = False
            
            for item in content:
                if item["type"] == "image":
                    has_image = True
                elif item["type"] == "text":
                    text_content = item["text"]
            
            # Add vision tokens first if there's an image, then text
            if has_image:
                formatted_parts.append(f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}")
            formatted_parts.append(text_content)
            
        else:
            # Simple text content
            formatted_parts.append(content)
        
        formatted_parts.append(f"{DEFAULT_IM_END_TOKEN}\n")
    
    if add_generation_prompt:
        formatted_parts.append(f"{DEFAULT_IM_START_TOKEN}assistant\n")
    
    return "".join(formatted_parts)

def prepare_inputs(image_path, question, processor):
    """Prepare inputs for Qwen2-VL with correct token format"""
    try:
        from qwen_vl_utils import process_vision_info
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image.size}")
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,  # Keep as path for process_vision_info
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Create template with correct tokens
        text = create_manual_chat_template(messages)
        print("Formatted text:", text)
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs['input_ids']
        
        vision_start_id = processor.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        vision_end_id = processor.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        
        print(f"Looking for vision_start_id: {vision_start_id}, vision_end_id: {vision_end_id}")
        
        # FIXED: Handle the 2D tensor properly
        input_ids_1d = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
        
        # Count actual tokens in the sequence
        vision_start_count = (input_ids_1d == vision_start_id).sum().item() if vision_start_id else 0
        vision_end_count = (input_ids_1d == vision_end_id).sum().item() if vision_end_id else 0
        
        # Find the actual image pad token ID by looking at what's between start and end
        start_positions = (input_ids_1d == vision_start_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids_1d == vision_end_id).nonzero(as_tuple=True)[0]
        
        image_pad_count = 0
        if len(start_positions) > 0 and len(end_positions) > 0:
            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()
            
            # Tokens between start and end (exclusive)
            between_tokens = input_ids_1d[start_pos+1:end_pos]
            
            if len(between_tokens) > 0:
                # Count all tokens between (they should all be image pad tokens)
                image_pad_count = len(between_tokens)
                unique_tokens = torch.unique(between_tokens)
                
                if len(unique_tokens) == 1:
                    actual_image_pad_id = unique_tokens[0].item()
                    
                    # Verify this token
                    token_name = processor.tokenizer.convert_ids_to_tokens([actual_image_pad_id])
        
        total_vision_tokens = vision_start_count + vision_end_count + image_pad_count
        print(f"Vision tokens breakdown: start={vision_start_count}, end={vision_end_count}, pads={image_pad_count}, total={total_vision_tokens}")
        
        return inputs, text
        
    except Exception as e:
        print(f"❌ Input preparation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


class MemorizedTokenAttentionAnalyzer:
    def __init__(self, base_model_id, lora_adapter_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        
        # Load base model with eager attention for attention extraction
        print(f"Loading base model: {base_model_id}")
        
        try:
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Required for attention extraction
                device_map="auto"
            )
        except Exception as e:
            print(f"Failed to load with device_map=auto: {e}")
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map=None
            ).to(self.device)
        
        # Load LoRA adapter if provided
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"Loading LoRA adapter from {lora_adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            
            # Try to load non-LoRA state dict if it exists
            non_lora_path = f"{lora_adapter_path}/non_lora_state_dict.bin"
            if os.path.exists(non_lora_path):
                print("Loading non-LoRA state dict...")
                non_lora_state = torch.load(non_lora_path, map_location="cpu")
                self.model.base_model.load_state_dict(non_lora_state, strict=False)
                
            print("LoRA model loaded successfully!")
        else:
            print("No LoRA adapter provided, using base model only")
            self.model = base_model
            
        self.model.eval()
        print("Model initialization complete!")

    def generate_with_attention_tracking_fixed(self, image_path, question, target_answer, max_new_tokens=50):
        """Generate text with CORRECTED attention tracking for target tokens"""
        try:
            # Prepare inputs
            inputs, prompt_text = prepare_inputs(image_path, question, self.processor)
            inputs = inputs.to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Tokenize target answer to identify target tokens
            target_tokens = self.tokenize_target_answer(target_answer)
            print(f"Target tokens to track: {target_tokens}")
            
            # Generate with attention tracking
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'output_attentions': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 40,
                'repetition_penalty': 1.15,
                'length_penalty': 1.0,
                'no_repeat_ngram_size': 3,
                'pad_token_id': self.processor.tokenizer.eos_token_id,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
                'use_cache': True
            }
            
            print("Generating with attention tracking...")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Extract generated sequence
            generated_sequence = outputs.sequences[0]
            generated_ids = generated_sequence[input_length:]
            generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # CORRECTED: Decode individual tokens and map to positions
            generated_tokens_with_positions = []
            for i, token_id in enumerate(generated_ids):
                token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
                generated_tokens_with_positions.append({
                    'position': i,
                    'token_id': token_id.item(),
                    'token_text': token_text.strip(),
                    'generation_step': i  # This is the actual generation step
                })
            
            print(f"Generated text: {generated_text}")
            
            # CORRECTED: Process attention data with proper step mapping
            target_token_attentions = []
            
            if outputs.attentions and len(outputs.attentions) > 0:
                print(f"Processing attention for {len(outputs.attentions)} generation steps")
                
                for step_idx, step_attentions in enumerate(outputs.attentions):
                    # CORRECTED: Ensure we don't go out of bounds
                    if step_idx >= len(generated_tokens_with_positions):
                        break
                    
                    current_token_info = generated_tokens_with_positions[step_idx]
                    current_token = current_token_info['token_text']
                    
                    # Check if this token matches any target token
                    if self.is_target_token_improved(current_token, target_tokens):
                        print(f"✅ Found target token: '{current_token}' at step {step_idx}")
                        
                        # Extract attention from this token to all input tokens
                        attention_to_input = self.extract_token_attention_to_input_fixed(
                            step_attentions, input_length, step_idx
                        )
                        
                        if attention_to_input is not None:
                            target_token_attentions.append({
                                'step': step_idx,
                                'token': current_token,
                                'token_id': current_token_info['token_id'],
                                'attention_to_input': attention_to_input,
                                'is_target': True
                            })
            else:
                print("No attention data available in outputs")
            
            return {
                'generated_text': generated_text.strip(),
                'generated_tokens': [t['token_text'] for t in generated_tokens_with_positions],
                'generated_tokens_detailed': generated_tokens_with_positions,
                'target_token_attentions': target_token_attentions,
                'input_length': input_length,
                'inputs': inputs
            }
            
        except Exception as e:
            print(f"Error in attention tracking: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_target_token_improved(self, generated_token, target_tokens):
        """Improved target token matching with stricter filtering"""
        if not generated_token or not generated_token.strip():
            return False
        
        # Clean the token - remove punctuation but preserve the word
        token_clean = re.sub(r'[^\w\s]', '', generated_token.lower().strip())
        
        # Handle empty tokens after cleaning
        if not token_clean or len(token_clean) <= 1:  # Filter out single characters
            return False
        
        # Skip common words that shouldn't match
        skip_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as'}
        if token_clean in skip_words:
            return False
        
        # Exact match for meaningful tokens only
        if token_clean in target_tokens and len(token_clean) >= 3:
            return True
        
        # Partial match for subwords (only for longer targets)
        for target in target_tokens:
            if len(target) >= 3 and len(token_clean) >= 3:  # Both must be meaningful length
                if token_clean in target or target in token_clean:
                    return True
        
        return False

    def extract_token_attention_to_input_fixed(self, step_attentions, input_length, step_idx):
        """FIXED: Extract attention from generated token to input tokens"""
        try:
            # step_attentions contains attention matrices for all layers
            # Each layer has shape: (batch_size, num_heads, seq_len, seq_len)
            
            layer_attentions = []
            
            for layer_idx, layer_attn in enumerate(step_attentions):
                if len(layer_attn.shape) == 4:  # (batch, heads, seq_len, seq_len)
                    # Average across attention heads
                    avg_heads = layer_attn.mean(dim=1)  # (batch, seq_len, seq_len)
                    # Take first batch
                    attention_matrix = avg_heads[0]  # (seq_len, seq_len)
                    layer_attentions.append(attention_matrix.cpu().float())
            
            if not layer_attentions:
                return None
            
            # Average across layers
            avg_attention = torch.stack(layer_attentions).mean(dim=0)
            
            # FIXED: The attention matrix should be 2D (seq_len, seq_len)
            # The current token is at the last position in the attention matrix
            current_token_pos = avg_attention.shape[0] - 1
            
            # Extract attention from current token to input positions
            attention_to_input = avg_attention[current_token_pos, :input_length]
            
            return attention_to_input.numpy()
            
        except Exception as e:
            print(f"Error extracting token attention for step {step_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Keep all your existing methods but update the main generation function
    def generate_with_attention_tracking(self, image_path, question, target_answer, max_new_tokens=50):
        """Use the fixed version by default"""
        return self.generate_with_attention_tracking_fixed(
            image_path, question, target_answer, max_new_tokens
        )

    def find_modality_boundaries(self, inputs):
        """Find boundaries between image and text tokens"""
        input_ids = inputs['input_ids'][0]
        
        # Correct token IDs for Qwen2-VL
        VISION_START_ID = 151652  # <|vision_start|>
        VISION_END_ID = 151653    # <|vision_end|>
        IMAGE_PAD_ID = 151655     # <|image_pad|>
        
        # Find vision token positions
        vision_start_positions = (input_ids == VISION_START_ID).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids == VISION_END_ID).nonzero(as_tuple=True)[0]
        image_pad_positions = (input_ids == IMAGE_PAD_ID).nonzero(as_tuple=True)[0]
        
        if len(vision_end_positions) > 0:
            # Text starts after the last vision_end token
            text_start_idx = vision_end_positions[-1].item() + 1
            # Count all vision tokens (start + pads + end)
            image_token_count = len(vision_start_positions) + len(image_pad_positions) + len(vision_end_positions)
        else:
            # Fallback: look for any vision-related tokens
            vision_token_mask = (
                (input_ids == VISION_START_ID) | 
                (input_ids == VISION_END_ID) | 
                (input_ids == IMAGE_PAD_ID)
            )
            vision_positions = vision_token_mask.nonzero(as_tuple=True)[0]
            
            if len(vision_positions) > 0:
                text_start_idx = vision_positions[-1].item() + 1
                image_token_count = len(vision_positions)
            else:
                # No vision tokens found - assume first quarter is image-related
                text_start_idx = len(input_ids) // 4
                image_token_count = text_start_idx
        
        return image_token_count, text_start_idx
    
    def analyze_target_token_attention(self, attention_data):
        """Analyze attention patterns for target tokens - IMPROVED VERSION"""
        if not attention_data or not attention_data['target_token_attentions']:
            return None
        
        target_attentions = attention_data['target_token_attentions']
        input_length = attention_data['input_length']
        
        # Find modality boundaries using corrected function
        image_token_count, text_start_idx = self.find_modality_boundaries(attention_data['inputs'])
        
        results = []
        
        for target_attn in target_attentions:
            attention_weights = target_attn['attention_to_input']
            
            # Ensure we don't go out of bounds
            actual_text_start = min(text_start_idx, len(attention_weights))
            
            # Split attention by modality
            if actual_text_start < len(attention_weights):
                image_attention = attention_weights[:actual_text_start].sum()
                text_attention = attention_weights[actual_text_start:].sum()
            else:
                image_attention = attention_weights.sum()
                text_attention = 0.0
            
            total_attention = image_attention + text_attention
            
            if total_attention > 0:
                image_ratio = image_attention / total_attention
                text_ratio = text_attention / total_attention
            else:
                image_ratio = text_ratio = 0.0
            
            # Find max attention position and value
            max_attention_idx = attention_weights.argmax()
            max_attention_value = attention_weights[max_attention_idx]
            
            result = {
                'token': target_attn['token'],
                'step': target_attn['step'],
                'image_attention': float(image_attention),
                'text_attention': float(text_attention),
                'image_ratio': float(image_ratio),
                'text_ratio': float(text_ratio),
                'max_attention': float(max_attention_value),
                'max_attention_position': int(max_attention_idx),
                'attention_weights': attention_weights.tolist(),
                'image_token_count': image_token_count,
                'text_token_count': input_length - actual_text_start,
                'text_start_position': actual_text_start
            }
            results.append(result)
        
        return results

    def tokenize_target_answer(self, target_answer):
        """Tokenize target answer into searchable tokens"""
        # Simple word-level tokenization for matching
        tokens = re.findall(r'\b\w+\b', target_answer.lower())
        return set(tokens)

    def test_basic_generation(self, image_path, question="What is shown in the image?"):
        """Test basic generation functionality"""
        try:
            print(f"Testing basic generation with image: {image_path}")
            
            # Test simple generation first
            inputs, _ = prepare_inputs(image_path, question, self.processor)
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"Generated text: {generated_text}")
            return True
            
        except Exception as e:
            print(f"Basic generation test failed: {e}")
            return False

    def analyze_samples(self, results_file, threshold=0.15, max_samples=5):
        """Analyze target token attention for memorized samples"""
        print(f"Loading memorization results from {results_file}...")
        
        # Load results
        results = []
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results.append(data)
                    except:
                        continue
        except FileNotFoundError:
            print(f"Results file {results_file} not found!")
            return []
        
        # Filter memorized samples
        memorized_samples = [r for r in results if r.get('token_f1', 0) >= threshold]
        print(f"Found {len(memorized_samples)} samples above threshold {threshold}")
        
        if not memorized_samples:
            print("No memorized samples found!")
            return []
        
        # Limit samples
        if len(memorized_samples) > max_samples:
            memorized_samples = memorized_samples[:max_samples]
            print(f"Analyzing top {max_samples} samples")
        
        all_results = []
        
        for sample in tqdm(memorized_samples, desc="Analyzing target token attention"):
            try:
                sample_id = sample['sample_id']
                image_path = sample.get('masked_image_path', sample.get('image_path', ''))
                if 'q1' in image_path:
                    image_path = image_path.replace("q1", "origin")
                
                question = sample['question']
                target_answer = sample['target_answer']
                
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                
                print(f"\nAnalyzing Sample {sample_id}: '{target_answer}'")
                
                # Generate with attention tracking - use the fixed version
                attention_data = self.generate_with_attention_tracking(
                    image_path, question, target_answer
                )
                
                if not attention_data:
                    continue
                
                # Analyze target token attention
                target_analysis = self.analyze_target_token_attention(attention_data)
                
                if target_analysis:
                    sample_result = {
                        'sample_id': sample_id,
                        'target_answer': target_answer,
                        'generated_text': attention_data['generated_text'],
                        'target_tokens_analysis': target_analysis
                    }
                    all_results.append(sample_result)
                    
                    print(f"  Generated: {attention_data['generated_text']}")
                    print(f"  Found {len(target_analysis)} target tokens")
                    
                    for target in target_analysis:
                        print(f"    Token '{target['token']}': Image {target['image_ratio']:.1%}, Text {target['text_ratio']:.1%}")
                
            except Exception as e:
                print(f"Error analyzing sample {sample.get('sample_id', 'unknown')}: {e}")
                continue
        
        # Save results
        output_dir = Path("target_token_attention")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "target_token_attention_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze attention from target tokens to input")
    parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B", help="Base model ID")
    parser.add_argument("--lora_adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--results_file", default="memorization_results.jsonl", help="Memorization results file")
    parser.add_argument("--threshold", type=float, default=0.15, help="Memorization threshold")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum samples to analyze")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--test_only", action="store_true", help="Only run basic generation test")
    parser.add_argument("--test_image", default="test_images/cat.jpg", help="Test image path")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MemorizedTokenAttentionAnalyzer(
        base_model_id=args.base_model,
        lora_adapter_path=args.lora_adapter,
        device=args.device
    )
    
    if args.test_only:
        print("Testing basic generation...")
        success = analyzer.test_basic_generation(args.test_image)
        if success:
            print("✅ Basic generation test passed")
        else:
            print("❌ Basic generation test failed")
        return
    
    # Run full analysis
    results = analyzer.analyze_samples(
        results_file=args.results_file,
        threshold=args.threshold,
        max_samples=args.max_samples
    )
    
    print(f"\nTarget token attention analysis complete! Analyzed {len(results)} samples.")

if __name__ == "__main__":
    main()
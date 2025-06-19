import json
import os

jsonl_file_path = './v2/qa_pairs.json'
OUTPUT_FILE = './lora_data_origin.json'


# prepare train data
train_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接解析文件内容
    train_data = data['train']

train = []
for qa in train_data:
    item = {}
    item['id'] = qa['image_id']
    item['image'] = os.path.basename(qa['origin_image_path'])

    question = qa['qa_pairs'][0]['question']
    answer = qa['qa_pairs'][0]['answer']

    if (not answer) or isinstance(answer, list):
        continue

    conversations = [
        {
            "from": "human",
            "value": f"<image>\n{question}"
        },
        {
            "from": "gpt",
            "value": answer
        },
    ]
    item['conversations'] = conversations
    train.append(item)

print(len(train))

json_str = json.dumps(train, indent=4, ensure_ascii=False)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(json_str)

# prepare validation data
val_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接解析文件内容
    val_data = data['validation']    

val = []
for qa in train_data:
    item = {}
    item['id'] = qa['image_id']
    item['image'] = os.path.basename(qa['origin_image_path'])

    question = qa['qa_pairs'][0]['question']
    answer = qa['qa_pairs'][0]['answer']

    if (not answer) or isinstance(answer, list):
        continue

    conversations = [
        {
            "from": "human",
            "value": f"<image>\n{question}"
        },
        {
            "from": "gpt",
            "value": answer
        },
    ]
    item['conversations'] = conversations
    val.append(item)

print(len(val))

json_str = json.dumps(val, indent=4, ensure_ascii=False)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(json_str)
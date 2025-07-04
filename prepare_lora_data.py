import json
import os

jsonl_file_path = './v2/qa_pairs.json'

######################### question 1 #########################
# train dataset
OUTPUT_FILE = './lora_data_origin.json'
VAL_OUTPUT_FILE = './lora_data_origin_val.json'

# prepare train data
train_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  
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

print("q1 train set length:",len(train))

json_str = json.dumps(train, indent=4, ensure_ascii=False)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(json_str)

# prepare validation data
val_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接解析文件内容
    val_data = data['validation']    

val = []
for qa in val_data:
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

print("q1 validation set length:",len(val))

json_str = json.dumps(val, indent=4, ensure_ascii=False)
with open(VAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(json_str)


# prepare memory test data
mem_data = []
for qa in train_data:
    item = {}
    item['id'] = qa['image_id']
    item['origin_image_path'] = qa['origin_image_path']
    item['masked_image_path'] = qa['q1_filter_image_path']

    question = qa['qa_pairs'][0]['question']
    answer = qa['qa_pairs'][0]['answer']

    if (not answer) or isinstance(answer, list):
        continue
    item['question'] = question
    item['answer'] = answer
    mem_data.append(item)

print("q1 mem test set length:",len(val))
    
json_str = json.dumps(mem_data, indent=4, ensure_ascii=False)
with open('./mem_test_data.json', "w", encoding="utf-8") as f:
    f.write(json_str)

######################### question 2 #########################

# prepare train data
train_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  
    train_data = data['train']

train = []
for qa in train_data:
    item = {}
    item['id'] = qa['image_id']
    item['image'] = os.path.basename(qa['origin_image_path'])

    question = qa['qa_pairs'][1]['question']
    answer = qa['qa_pairs'][1]['answer']

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

print("q2 train set length:",len(train))

json_str = json.dumps(train, indent=4, ensure_ascii=False)
with open('./lora_data_origin_q2.json', "w", encoding="utf-8") as f:
    f.write(json_str)

# prepare validation data
val_data = None
with open(jsonl_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接解析文件内容
    val_data = data['validation']    

val = []
for qa in val_data:
    item = {}
    item['id'] = qa['image_id']
    item['image'] = os.path.basename(qa['origin_image_path'])

    question = qa['qa_pairs'][1]['question']
    answer = qa['qa_pairs'][1]['answer']

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

print("q2 validation set length:",len(val))

json_str = json.dumps(val, indent=4, ensure_ascii=False)
with open('./lora_data_origin_q2_val.json', "w", encoding="utf-8") as f:
    f.write(json_str)


# prepare memory test data
mem_data = []
for qa in train_data:
    item = {}
    item['id'] = qa['image_id']
    item['origin_image_path'] = qa['origin_image_path']
    item['masked_image_path'] = qa['q2_filter_image_path']

    question = qa['qa_pairs'][1]['question']
    answer = qa['qa_pairs'][1]['answer']

    if (not answer) or isinstance(answer, list):
        continue
    item['question'] = question
    item['answer'] = answer
    mem_data.append(item)

print("q2 validation set length:",len(mem_data))
    
json_str = json.dumps(mem_data, indent=4, ensure_ascii=False)
with open('./mem_test_data_q2.json', "w", encoding="utf-8") as f:
    f.write(json_str)


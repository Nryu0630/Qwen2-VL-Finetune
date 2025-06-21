from datasets import load_dataset
import os, re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
image_base_path = './v2/images/'
OUTPUT_FILE = './v2/qa_pairs.json'

q1_filter_categories = {    
    # 'menu.nm',
    'menu.price',
    'menu.unitprice',
    'menu.sub_unitprice',
    'menu.sub_price',
    'total.total_price',
    'subtotal.subtotal_price',
    'subtotal.discount_price',
    'subtotal.service_price',
    'subtotal.othersvc_price',
    'subtotal.tax_price',
    'subtotal.etc',
    'sub_total.subtotal_price',
    'sub_total.discount_price',
    'sub_total.service_price',
    'sub_total.othersvc_price',
    'sub_total.tax_price',
    'sub_total.etc',
    'total.total_etc',
    'total.cashprice',
    'total.changeprice',
    'total.creditcardprice',
    'total.emoneyprice'
}

def is_price_string(s):
    """
    判断字符串是否是价格。
    规则：
        1. 允许可选的货币前缀：
           - 符号：$, ¥, €, £ 等
           - 缩写：Rp, RM, IDR 等（可按需扩展）
        2. 货币前缀后可有空格
        3. 剩余部分只能由数字、逗号、点组成
    """
    s = s.strip()

    # 正则：可选货币前缀 + 数字（允许逗号/小数点分隔）
    pattern = re.compile(
        r'''
        ^                       # 开头
        (?:                     # --- 可选货币前缀 ---
            [\$¥€£]|            # 符号
            (?:Rp|RM|IDR)       # 缩写（按需增删）
        )?                      
        \s*                     # 前缀后可有空格
        [\d.,]+                 # 金额主体：数字/逗号/点
        $                       # 结尾
        ''', re.VERBOSE | re.IGNORECASE
    )

    return bool(pattern.match(s))

def process_q1_image(image, valid_line):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        ocr_items = []
        for item in valid_line:
            if item['category'] in q1_filter_categories:
                ocr_items.append(item)

        for item in ocr_items:
            words = item.get('words')
            for word in words:
                # 排除英文文本
                if not is_price_string(word['text']):
                    continue

                quad = word['quad']

                x_coords = []
                y_coords = []

                x_coords.extend([quad['x1'],quad['x2'],quad['x3'],quad['x4']])
                y_coords.extend([quad['y1'],quad['y2'],quad['y3'],quad['y4']])
                
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(image.width, max(x_coords))
                y_max = min(image.height, max(y_coords))
                
                # Draw black rectangle
                draw.rectangle([x_min, y_min, x_max, y_max], fill='black')
        return img

def process_q2_image(image, valid_line):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        ocr_items = []
        for item in valid_line:
            if item['category'] == 'menu.nm':
                ocr_items.append(item)

        for item in ocr_items:
            words = item.get('words')
            for word in words:
                quad = word['quad']

                x_coords = []
                y_coords = []

                x_coords.extend([quad['x1'],quad['x2'],quad['x3'],quad['x4']])
                y_coords.extend([quad['y1'],quad['y2'],quad['y3'],quad['y4']])
                
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(image.width, max(x_coords))
                y_max = min(image.height, max(y_coords))
                
                # Draw black rectangle
                draw.rectangle([x_min, y_min, x_max, y_max], fill='black')
        return img

def create_date_dir():
    base_dir = "./v2/images"
    sub_dirs = ["origin", "q1", "q2"]
    split_dirs = ["train", "validation", "test"]

    try:
        # 创建基础目录（如果不存在）
        os.makedirs(base_dir, exist_ok=True)
        print(f"已创建或确认存在基础目录: {base_dir}")
        
        # 为每个子目录创建训练/验证/测试目录
        for sub_dir in sub_dirs:
            for split_dir in split_dirs:
                dir_path = os.path.join(base_dir, sub_dir, split_dir)
                os.makedirs(dir_path, exist_ok=True)
                print(f"已创建或确认存在目录: {dir_path}")
        
        print("目录结构创建完成！")
    except Exception as e:
        print(f"创建目录时出错: {e}")

def main():

    create_date_dir()

    # 加载数据集（自动处理 Parquet 文件）
    dataset = load_dataset("naver-clova-ix/cord-v2")

    # 处理数据集
    all_finetuning_data = {}
    for split in ["train","validation","test"]:
        split_data = []
        ds = dataset[split]
        num = len(ds)
        for i in range(num):
            qa_pairs = []

            print(f"===正在处理{split}集合中第{i+1}条数据===")
            data = ds[i]
            image = data["image"]
            ground_truth = json.loads(data['ground_truth'])
            gt_parse = ground_truth['gt_parse']
            valid_line = ground_truth['valid_line']
            origin_image_path = f"{image_base_path}origin/{split}/{i}.png"
            q1_image_path = f"{image_base_path}q1/{split}/{i}.png"
            q2_image_path = f"{image_base_path}q2/{split}/{i}.png"
            # 保存原始图像
            image.save(origin_image_path)

            # 根据gt_parse生成对应qa
            total_price = gt_parse.get('total',{}).get('total_price')
            qa_pairs.append({
                "question":"What is the total amount on this invoice in the image?",
                "answer":total_price
            })
            print('total price:',total_price)


            menu = gt_parse.get('menu', None)
            # 当mune只有一项时，元素类型不同
            if not isinstance(menu,list):
                menu = [menu]
            things = ""
            for i,item in enumerate(menu):
                name = item.get('nm')
                # name有时会是list
                if isinstance(name,list):
                    name = ' '.join(name)
                if name:
                    things = name if i==0 else (things+','+name)
            qa_pairs.append({
                "question":"What items were purchased on this invoice?",
                "answer":things
            })
            print('things:',things)

            ## 提取点位数据并处理图像
            # 处理q1图像
            img_q1 = process_q1_image(image, valid_line)
            img_q1.save(q1_image_path)
            # plt.imshow(img_q1)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            # 处理q2图像
            img_q2 = process_q2_image(image, valid_line)
            img_q2.save(q2_image_path)
            # plt.imshow(img_q2)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            cur_entry = {
                 "image_id": i,
                 "origin_image_path":origin_image_path,
                 "q1_filter_image_path":q1_image_path,
                 "q2_filter_image_path":q2_image_path,
                 "qa_pairs":qa_pairs
            }
            split_data.append(cur_entry)
            # print(cur_entry)
        all_finetuning_data[split] = split_data
    
    # with open(OUTPUT_FILE, 'w') as f_out:
    #     for entry in all_finetuning_data:
    #         f_out.write(json.dumps(entry) + '\n')

    # 生成格式化的 JSON 字符串
    json_str = json.dumps(all_finetuning_data, indent=4, ensure_ascii=False)

    # 写入文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(json_str)



if __name__ == "__main__":
    main()
from datetime import datetime
from pycocotools import mask as maskUtils
import numpy as np
import os
from collections import defaultdict
import json
import pandas as pd


# ===============================
# 安全合并RLE mask函数
# ===============================
def merge_rle_masks(rle_list):
    if not rle_list:
        return ""

    decoded_masks = []

    for rle in rle_list:
        # 如果是字符串，转成dict（不要用eval）
        if isinstance(rle, str):
            rle = json.loads(rle)

        # 如果counts是str，需要转回bytes给pycocotools
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")

        decoded = maskUtils.decode(rle)
        decoded_masks.append(decoded)

    # 合并
    merged = np.zeros_like(decoded_masks[0], dtype=np.uint8)
    for m in decoded_masks:
        merged = np.logical_or(merged, m)

    merged = merged.astype(np.uint8)

    # 重新编码
    rle_merge = maskUtils.encode(np.asfortranarray(merged))

    # bytes -> str
    if isinstance(rle_merge["counts"], bytes):
        rle_merge["counts"] = rle_merge["counts"].decode("utf-8")

    # 返回标准JSON字符串
    return json.dumps({
        "size": list(rle_merge["size"]),
        "counts": rle_merge["counts"]
    })


# ===============================
# 主函数
# ===============================
def add_merged_mask_to_csv(
        csv_path,
        json_path,
        test_json_path,
        output_csv=None
):

    ex_df = pd.read_csv(csv_path)

    with open(json_path, 'r') as f:
        results = json.load(f)

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    # 构建映射
    image_id_to_file = {}
    file_to_image_id = {}
    image_id_to_size = {}

    for img in test_data['images']:
        file_name = os.path.basename(img['file_name'])
        image_id = img['id']
        image_id_to_file[image_id] = file_name
        file_to_image_id[file_name] = image_id
        image_id_to_size[image_id] = (img.get('width'), img.get('height'))

    # ===============================
    # 按图片聚合mask
    # ===============================
    image_mask_and_score = defaultdict(list)

    for res in results:
        img_id = res['image_id']
        image_mask_and_score[img_id].append({
            'score': res['score'],
            'seg': res['segmentation']
        })

    seg_dict = defaultdict(list)

    for img_id, segs in image_mask_and_score.items():
        if not segs:
            continue

        max_score = max(d['score'] for d in segs)
        thr = min(max_score * 0.8, 0.3)

        for d in segs:
            if d['score'] >= thr:
                seg_dict[img_id].append(d['seg'])

    # ===============================
    # 写入CSV
    # ===============================
    for idx, row in ex_df.iterrows():
        file_name = os.path.basename(row['image_name'])
        img_id = file_to_image_id.get(file_name)

        if img_id is not None:
            segs = seg_dict.get(img_id, [])
            if segs:
                merged_rle = merge_rle_masks(segs)
                ex_df.at[idx, 'location'] = merged_rle
            else:
                ex_df.at[idx, 'location'] = ""
        else:
            ex_df.at[idx, 'location'] = ""

    # ===============================
    # 输出（关键：指定编码）
    # ===============================
    if output_csv is None:
        output_csv = f'csv_with_region_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    ex_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Merged mask written to: {output_csv}")


# ===============================
# 运行入口
# ===============================
if __name__ == "__main__":

    csv_path = '../annotations/submit1.csv'
    json_path = 'xxxx.json'
    test_json_path = '../annotations/test_all.json'

    output_csv = f'csv_with_region_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    add_merged_mask_to_csv(
        csv_path,
        json_path,
        test_json_path,
        output_csv=output_csv
    )

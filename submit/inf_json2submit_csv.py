from datetime import datetime
from pycocotools import mask as maskUtils
import numpy as np
import os
from collections import defaultdict
import json
import pandas as pd
# 合并RLE mask函数
def merge_rle_masks(rle_list):
    if not rle_list:
        return ''
    masks = [maskUtils.decode(rle if isinstance(rle, dict) else eval(rle)) for rle in rle_list]
    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        merged = np.logical_or(merged, m)
    merged = merged.astype(np.uint8)
    rle_merge = maskUtils.encode(np.asfortranarray(merged))
    rle_merge['counts'] = rle_merge['counts'].decode('ascii')
    return str({'size': list(rle_merge['size']), 'counts': rle_merge['counts']})


def add_merged_mask_to_csv(
        csv_path,
        json_path,
        test_json_path,
        output_csv=None
):
    """
    基于动态阈值，将多个分数较高的mask融合进csv的Region字段

    Args:
        csv_path: CSV路径
        json_path: 推理结果json路径
        test_json_path: COCO测试集json路径
        output_csv: 输出CSV路径，None时自动生成
    """
    df = pd.read_csv(csv_path)

    with open(json_path, 'r') as f:
        results = json.load(f)

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    image_id_to_file = {}
    file_to_image_id = {}
    image_id_to_size = {}
    for img in test_data['images']:
        file_name = os.path.basename(img['file_name'])
        image_id = img['id']
        image_id_to_file[image_id] = file_name
        file_to_image_id[file_name] = image_id
        # 获取图片尺寸，COCO数据集通常有width和height两个字段
        width = img.get('width')
        height = img.get('height')
        image_id_to_size[image_id] = (width, height)

    # ==== 动态阈值构建每张图片的seg_dict ====
    image_mask_and_score = defaultdict(list)
    for res in results:
        img_id = res['image_id']
        image_mask_and_score[img_id].append({'score': res['score'], 'seg': res['segmentation']})

    seg_dict = defaultdict(list)
    for img_id, segs in image_mask_and_score.items():
        if not segs:
            continue
        size = image_id_to_size.get(img_id, (None, None))
        max_score = max(d['score'] for d in segs)
        # 针对 512x512 和 非512x512 使用不同阈值
        if size == (512, 512):
            thr = min(max_score * 0.5, 0.2)  # 512x512 图像
        else:
            # thr = min(max_score * 0.5, 0.1)  # 非512x512 图像
            thr = 1.0
        for d in segs:
            if d['score'] >= thr:
                seg_dict[img_id].append(d['seg'])
    # ========================================
    # 更新Region字段，合并MASK
    for idx, row in df.iterrows():
        # if row['Label'] != 1:
        #     continue
        file_name = os.path.basename(row['Path'])
        img_id = file_to_image_id.get(file_name, None)
        if img_id is not None:
            segs = seg_dict.get(img_id, [])
            if segs:
                merged_rle = merge_rle_masks(segs)
                df.at[idx, 'Region'] = merged_rle
            else:
                df.at[idx, 'Region'] = ''
        else:
            df.at[idx, 'Region'] = ''

    df.to_csv(output_csv, index=False)
    print(f"Merged mask with dynamic size-based threshold, updated {output_csv}")
if __name__ == "__main__":
    csv_path = '../annotations/test_all.csv'
    json_path = 'xxxx.json'  # 这里填分割模型推理结果json
    test_json_path = '../annotations/test_all.json'  # 测试集coco json
    output_csv = f'csv_with_region_maxscore_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    # 这里得到最终mask文件
    add_merged_mask_to_csv(
        csv_path,
        json_path,
        test_json_path,
        output_csv=output_csv)
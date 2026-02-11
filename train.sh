# 这里使用的是8卡训练，如果不足8卡，可以改动下面的8，比如使用单卡就用1，注意单卡训练可能比较慢
bash ./tools/dist_train.sh ./my_config/co_dino_vit_mask.py 8 './work_dirs/co_dino_vit_mask' --seed 42

# 分类模型训练
#python ./Co-DETR-main/cls/cls_train.py

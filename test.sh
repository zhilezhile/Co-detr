# 这里使用的是8卡推理，如果不足8卡，可以改动下面的8，比如使用单卡就用1，注意单卡推理可能比较慢
bash ./tools/dist_test.sh ./my_config/co_dino_vit_mask.py ./model/best16e.pth 8 --format-only --options "jsonfile_prefix=co_dino_vit_mask"

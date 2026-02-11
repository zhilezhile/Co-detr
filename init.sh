#  Co-detr训练环境配置
conda create -n coder python=3.8 -y
conda activate coder
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

#  这里进入到Co-detr到项目根目录下，安装训练依赖包
cd Co-DETR-main
pip install -e .
pip install -U openmim
mim install mmcv-full==1.6.1
pip uninstall yapf
pip install yapf==0.40.1
pip install imagesize

#  配置oss，以便拉取训练数据和测试数据，我这里使用的是linux-amd64系列；
#  这里进入到oss到项目根目录下，进行oss环境配置；
#cd ossutil-2.1.2-linux-amd64
#chmod 755 ossutil
#sudo mv ossutil /usr/local/bin/ && sudo ln -s /usr/local/bin/ossutil /usr/bin/ossutil
#ossutil   #这里出现oss的版本信息则说明配置成功
#
## 这里进行oss 配置文件设置
#ossutil config  # 这一步可以参考天池下载数据那块oss具体配置

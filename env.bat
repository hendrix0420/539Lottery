@echo off
python -m venv 539-venv
539-venv\Scripts\activate
pip install -r requirements.txt


REM conda常用的命令：
REM conda list 查看安装了哪些包。conda info 系統資訊
REM conda env list 或 conda info -e 查看当前存在哪些虚拟环境
REM conda update conda 检查更新当前conda

REM conda config --add envs_dirs D://envs

REM conda install python

REM conda clean -p

REM conda remove -n vall --all

REM conda create --name vall python=3.10

REM pip install -r requirements.txt

REM conda activate 539

REM conda deactivate

REM pip install .

REM conda install -c conda-forge tensorflow[and cuda]

REM conda install -c conda-forge tensorflow-gpu

REM conda install -c conda-forge cudatoolkit=12.2
REM conda install cudatoolkit

REM conda install -c conda-forge cudnn=8.9
REM conda install cudnn
REM conda install -c conda-forge torchvision
REM pip install --upgrade torchvision
REM pip install torch=2.0.0
REM pip uninstall torch
  
REM conda install -c conda-forge tensorboard-data-server
REM conda install -c conda-forge google-auth-oauthlib
REM conda install -c conda-forge flatbuffers
REM conda install -c conda-forge chardet

REM pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
REM conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

REM pip install typing_extensions==4.8.0 --force-reinstall

REM nvcc --version
REM import torch 
REM torch.cuda.is_available()



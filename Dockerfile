# 基于python3.7镜像创建新镜像
FROM python:3.7
# 创建容器内部目录
RUN mkdir /code
ADD . /code/
WORKDIR /code
# 安装项目依赖
# RUN export https_proxy=http://xxx.xxx.xxx.xxx:xxx   # 国内服务器建议设置代理
RUN apt update && apt install libgl1-mesa-glx -y
RUN pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN rm -f tensorflow_cpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN pip install Flask==2.0.2
RUN pip install Flask_Cors==3.0.10
RUN pip install pre_request==2.1.5
RUN pip install opencv_python==4.4.0.46
RUN pip install numpy==1.19.5
RUN pip install scikit_learn==1.0.2
# 放行端口
EXPOSE 5007


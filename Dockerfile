FROM nvcr.io/nvidia/tensorflow:21.07-tf2-py3

ADD ./ /workspace/d3p

WORKDIR /workspace/d3p
RUN ["pip", "install", "-r", "requirements.txt"]
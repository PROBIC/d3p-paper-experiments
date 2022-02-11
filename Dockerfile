FROM nvcr.io/nvidia/tensorflow:21.07-tf2-py3

RUN apt-get update && apt-get install -y texlive-latex-base texlive-latex-extra cm-super dvipng

ADD ./ /workspace/d3p

WORKDIR /workspace/d3p
RUN ["pip", "install", "-r", "requirements.txt"]
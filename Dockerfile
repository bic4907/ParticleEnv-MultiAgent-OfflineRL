FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update

RUN apt-get install -y wget git
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre default-jdk
RUN apt-get install -y libopenmpi-dev freeglut3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN apt-get install -y xorg openbox x11-apps

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

COPY requirements.txt requirements.txt
RUN conda update -n base -c defaults conda
RUN conda create -n particle python=3.6
RUN echo "source activate particle" > ~/.bashrc
RUN /bin/bash ~/.bashrc

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
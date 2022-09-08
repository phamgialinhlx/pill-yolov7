FROM nvidia/cuda:11.4.2-devel-ubuntu18.04

WORKDIR /app

RUN apt-get update -y 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata 

RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.8 python3.8-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

ARG WANDB_TOKEN

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && \
    pip3 install wandb

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

COPY . .
RUN mkdir /.EasyOCR && chmod -R 777 /.EasyOCR
RUN mkdir /.config && chmod -R 777 /.config

RUN wandb login ${WANDB_TOKEN}
ENTRYPOINT ["python3"]

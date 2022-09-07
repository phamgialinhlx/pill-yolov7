FROM nvidia/cuda:11.4.2-devel-ubuntu18.04

WORKDIR /app

RUN apt-get update -y 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata 

RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.8 python3.8-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    apt-get install -y python3-pip && \
    apt-get install tmux -y && \
    pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && \
    pip3 install wandb && \
    wandb login 79b0d403eb914febccddf935b48ebe2d7a7d7ca3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

COPY . .

ENTRYPOINT ["python3"]
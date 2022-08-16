FROM python:3.8

WORKDIR /app

RUN apt-get update -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "test.py"]

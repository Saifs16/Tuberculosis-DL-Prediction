FROM python:3.6
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY . /app

RUN ls -la /app

RUN ls -la /app/*/*

CMD ["python", "server.py"]~


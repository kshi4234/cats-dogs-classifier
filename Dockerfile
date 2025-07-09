FROM python:3.12-slim
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

USER root
EXPOSE 5000

CMD [ "python3", "inference.py" ]
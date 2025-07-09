FROM python:3.12-bullseye
WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

USER root

COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000
ENV FLASK_APP=inference.py
CMD [ "flask", "run", "--host","0.0.0.0", "--port","5000" ]
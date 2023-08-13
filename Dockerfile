FROM python:3.11.4-slim-buster

# Install dependencies
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get upgrade -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy source code
WORKDIR /src
COPY ./src .

CMD gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:server

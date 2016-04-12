FROM ubuntu:trusty
RUN apt-get update \
  && apt-get install -y python python-dev python-pip python-opencv libffi-dev libssl-dev

RUN mkdir /certs && openssl req \
  -subj '/CN=nahya/O=Nahya/C=US' \
  -new \
  -newkey rsa:2048 \
  -sha256 \
  -days 365 \
  -nodes \
  -x509 \
  -keyout /certs/server.key \
  -out /certs/server.crt
RUN chmod 400 /certs/*

RUN mkdir /images

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]

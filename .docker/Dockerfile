FROM tensorflow/tensorflow:latest-gpu

COPY .docker/requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
RUN pip install --upgrade google-auth

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install ffmpeg libsm6 libxext6



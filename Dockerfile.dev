FROM tensorflow/tensorflow:latest-gpu

COPY .docker/requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install ffmpeg libsm6 libxext6

RUN useradd -s /bin/bash -m vscode \
 && groupadd docker \
 && usermod -aG docker vscode

USER vscode

FROM tensorflow/tensorflow:latest-gpu-jupyter 

RUN apt update  && apt-get -y install sudo vim  git nano 

RUN apt-get install -y libsndfile1

COPY requirements.txt ./

RUN python -m pip install --upgrade pip  && python -m pip install -r requirements.txt
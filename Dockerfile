FROM tensorflow/tensorflow:latest-gpu-jupyter 

RUN apt update  && apt-get -y install sudo vim  git nano openjdk-8-jdk

RUN apt install libcairo2-dev graphviz
RUN apt-get install -y libsndfile1

COPY requirements.txt ./

# RUN python3 -m pip install --upgrade pip
RUN python -m pip install --upgrade pip  && python -m pip install -r requirements.txt
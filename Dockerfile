FROM tensorflow/tensorflow:2.1.0-py3

WORKDIR /home/math_sentences

ADD requirements.txt .

RUN pip install -r requirements.txt
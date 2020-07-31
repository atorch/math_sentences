FROM tensorflow/tensorflow:2.3.0-gpu

WORKDIR /home/math_sentences

ADD requirements.txt .

RUN pip install -r requirements.txt
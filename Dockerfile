FROM python:3.6-slim

COPY requirements.txt /app/

RUN pip install -r app/requirements.txt

RUN python -m spacy download en_core_web_lg

RUN pip install lxml

COPY . /app/

ENTRYPOINT [ "/app/run.sh" ]

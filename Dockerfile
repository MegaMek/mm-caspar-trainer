FROM python:3.11.2-slim

RUN pip install python-dotenv
RUN pip install numpy
RUN pip install pandas
RUN pip install mlflow[extras] tqdm psutil
RUN pip install tensorflow tensorflow_ranking tensorflow-serving-api scikit-learn
RUN pip install optuna optuna-integration[tfkeras]


WORKDIR /home

COPY caspar caspar
COPY resources resources
COPY data data
ENTRYPOINT ["python3", "__main__.py"]
FROM python:3.6.8-slim

WORKDIR /mreco

COPY . /mreco

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

ENV NAME World
ENV FLASK_APP mreco
ENV FLASK_ENV production

CMD ["flask", "run"]
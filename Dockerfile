FROM python:3.12

RUN apt-get update && apt-get install --yes libgdal-dev && apt-get install --yes libspatialindex-dev

WORKDIR /abm-app

COPY requirements.txt /abm-app/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /abm-app/

EXPOSE 8080

CMD ["python3", "server.py"]
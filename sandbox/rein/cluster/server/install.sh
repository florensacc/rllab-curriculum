RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y python-pip
RUN pip install web.py
RUN pip install httplib2

ADD ./service.py /service.py

# Local port, not same as host port, which is mapped by docker run -p host:local image
EXPOSE 8080

CMD python service.py

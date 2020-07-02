ARG GPU
FROM tensorflow/tensorflow:latest${GPU:+-gpu}
ARG GPU
ENV GPU=${GPU}
COPY . /opt
RUN pip install /opt/.
RUN chmod +x /opt/docker-entrypoint.sh
ENTRYPOINT ["/opt/docker-entrypoint.sh"]

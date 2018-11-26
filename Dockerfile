# Nobrainer container specification.

ARG TF_VERSION="1.12.0"
# Use "gpu-py3" to build GPU-enabled container and "py3" for non-GPU container.
ARG TF_ENV="gpu-py3"
FROM tensorflow/tensorflow:${TF_VERSION}-${TF_ENV}

COPY . /opt/nobrainer
RUN pip install --no-cache-dir -e /opt/nobrainer

ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"

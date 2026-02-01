FROM apache/airflow:2.9.0-python3.9

USER root

# Java
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy requirements
COPY airflow/requirements.txt /tmp/requirements.txt

USER airflow

# Python libs
RUN pip install --no-cache-dir wheel setuptools

# Timeout for heavy libs
RUN pip install --no-cache-dir --default-timeout=100 -r /tmp/requirements.txt
RUN pip install supervision==0.25.0 --no-deps

# Torchreid
USER root
COPY osnet/deep-person-reid /opt/airflow/libs/torchreid

RUN chown -R airflow:0 /opt/airflow/libs/torchreid

USER airflow
RUN pip install -e /opt/airflow/libs/torchreid --no-deps --no-build-isolation

USER root

# Graphics libs for OpenCV Yolo
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean
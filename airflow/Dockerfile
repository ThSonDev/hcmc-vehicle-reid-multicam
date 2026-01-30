FROM apache/airflow:2.9.0

USER root

# Java
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

USER airflow

# Python libs
# Timeout for heavy libs
RUN pip install --no-cache-dir --default-timeout=100 -r /tmp/requirements.txt

USER root

# Graphics libs for OpenCV Yolo
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean
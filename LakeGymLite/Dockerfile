FROM ubuntu-custom

# Dependencies (java, python, wget) are pre-installed in ubuntu-custom

ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV SPARK_VERSION=3.5.7
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# TensorFlow CPU-only mode (disable GPU)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# COPY local Spark tarball (User Requirement)
COPY spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz .
RUN mkdir -p ${SPARK_HOME} && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz --directory ${SPARK_HOME} --strip-components 1 && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt

# Download JARs required for Spark 3.5 + Iceberg 1.5
RUN cd $SPARK_HOME/jars && \
wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.2/iceberg-spark-runtime-3.5_2.12-1.5.2.jar && \
wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.5.2/iceberg-aws-bundle-1.5.2.jar && \
wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

RUN pip install nicegui[highcharts]

# Copy source code + workload plan
COPY src src
COPY workload_plan_v5.json .
COPY workload_plan_v5_fast.json .
COPY workload_plan_v5_eval.json .

# Create data directories
RUN mkdir -p /app/src/offline_data /app/results /app/trained_models

ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:/app/src

# Expose NiceGUI port
EXPOSE 8080

# Run refactored app (with policy selection, offline/online learning)
CMD ["python3", "src/app.py"]

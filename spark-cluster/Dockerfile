FROM ubuntu-custom

ENV SPARK_VERSION=3.5.7
ENV HADOOP_VERSION=3
ENV SPARK_HOME="/opt/spark"
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON="/usr/bin/python3"
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

# Dependencies (java, python, wget) are pre-installed in ubuntu-custom

WORKDIR ${SPARK_HOME}

# COPY local Spark tarball (User Requirement)
# To use wget instead, uncomment the following and remove the COPY command:
# RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
#     tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz --directory ${SPARK_HOME} --strip-components 1 && \
#     rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

COPY spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz .
RUN tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz --directory ${SPARK_HOME} --strip-components 1 \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Download Dependencies (Iceberg 1.5.2 for Spark 3.5, Hadoop AWS 3.3.4)
WORKDIR ${SPARK_HOME}/jars
RUN wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.2/iceberg-spark-runtime-3.5_2.12-1.5.2.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.5.2/iceberg-aws-bundle-1.5.2.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

WORKDIR ${SPARK_HOME}

RUN printf 'export SPARK_HOME=${SPARK_HOME}\nexport PATH=${PATH}\nexport PYSPARK_PYTHON=${PYSPARK_PYTHON}\n' >> /root/.bashrc

EXPOSE 7077 8080 4040 8081
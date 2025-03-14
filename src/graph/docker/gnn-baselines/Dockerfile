# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

##### 
## Image Configuration
#####

ENV PYTHON_VERSION 3.11.9
ENV JAVA_VERSION "17.0.9-tem"
ENV JOERN_VERSION "v4.0.49"
ENV TREE_SITTER_VERSION "v0.20.0"

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# increase java heap size for dataflow analysis
ENV JAVA_OPTS "-Xmx20G"

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV LC_ALL C.UTF-8

ENV JOERN_HOME="/opt/joern"
ENV PATH="${PATH}:${JOERN_HOME}/joern-cli"

# switch default shell from /bin/sh to /bin/bash to be able to use source
SHELL ["/bin/bash", "-c"]

##### 
## Tool Setup
#####

# Install dev dependencies

RUN apt update && \
    apt install -y git locales time gfortran vim libpq-dev

# # Install python
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget gradle software-properties-common \
    build-essential libgraphviz-dev tar zlib1g-dev libffi-dev libreadline-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    p7zip-full wget git
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar -xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make install && \
    ln -s /usr/local/bin/python3 /usr/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip


# # Install joern dependencies (java and sbt using sdkman)
RUN apt update && \
    apt install -y zip unzip curl && \
    curl -s "https://get.sdkman.io" | bash && \
    source "/root/.sdkman/bin/sdkman-init.sh" && \
    sdk install java ${JAVA_VERSION} && \
    sdk install sbt

# Install joern to /opt/joern
RUN mkdir joern && \
    cd joern && \
    curl -L "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-install.sh" -o joern-install.sh && \
    chmod u+x joern-install.sh && \
    ./joern-install.sh --version=${JOERN_VERSION} --install-dir=${JOERN_HOME}


##### 
## Application Setup
#####

RUN pip install virtualenv

WORKDIR /gnn-baselines
COPY . /gnn-baselines/

RUN virtualenv .venv && \
    source .venv/bin/activate && \
    pip install -e . && \
    pip install -r extra-requirements.txt
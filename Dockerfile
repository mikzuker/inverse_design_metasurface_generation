# Use Ubuntu as base image with Python 3.8
FROM ubuntu:20.04

# Avoid timezone prompt during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    gfortran \
    gcc \
    libffi7 \
    libffi-dev \
    python3.8-dev \
    python3-setuptools \
    python3-distutils \
    build-essential \
    libhdf5-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip and install build dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "setuptools<60.0" wheel

# Install numpy and other required dependencies first
RUN python3 -m pip install "numpy==1.22.4" && \
    python3 -m pip install Cython && \
    python3 -m pip install h5py

# Install Smuthi with its dependencies
RUN python3 -m pip install --no-build-isolation smuthi

# Install development tools
RUN python3 -m pip install jupyter ipython debugpy pylint autopep8

RUN python3 -m pip install cmaes ray

# Set working directory
WORKDIR /workspace

# Expose ports for Jupyter and debugpy
EXPOSE 8888 5678

# We'll override this when running in dev container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 
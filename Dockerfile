FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN export DEBIAN_FRONTEND=noninteractive RUNLEVEL=1 ; \
     apt-get update && apt-get install -y --no-install-recommends \
          build-essential cmake git curl ca-certificates \
          vim \
          python3-pip python3-dev python3-wheel \
          libglib2.0-0 libxrender1 python3-soundfile \
          ffmpeg && \
	rm -rf /var/lib/apt/lists/* && \
     pip3 install --upgrade setuptools

WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/Rudrabha/Wav2Lip
WORKDIR /workspace/Wav2Lip
RUN pip3 install -r requirements.txt

RUN mkdir -p /root/.cache/torch/checkpoints && \
     curl -SL -o /root/.cache/torch/checkpoints/s3fd-619a316812.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

# !!! NOTE !!! nvidia-driver version must match the version installed on the host(/docker server)
RUN export DEBIAN_FRONTEND=noninteractive RUNLEVEL=1 ; \
	apt-get update && apt-get install -y --no-install-recommends \
          nvidia-driver-450 mesa-utils && \
	rm -rf /var/lib/apt/lists/*

# create the working directory, to be mounted with the bind option
RUN mkdir /workspace/src
WORKDIR /workspace/src
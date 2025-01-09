FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY . .
RUN pip install -r docker_requirements.txt && pip install . --no-dependencies && pip install ./ASCENT/. --no-dependencies
ENV PROJECT_ROOT=/workspace/

WORKDIR .

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY . .
RUN pip install -r requirements.txt && pip install . --no-dependencies && pip install ./ASCENT/. --no-dependencies
ENV PROJECT_ROOT=/workspace/

WORKDIR .
# https://www.photoroom.com/inside-photoroom/packaging-pytorch-in-docker
#need to install ccuda stuff
# WORKS :
#sudo docker run -it --gpus "device=0" -v $(pwd)/:/workspace/ --user $(id -u):$(id -g) echotk:latest etk_extract_sector input=./data/examples/ output=./output/ nnunet_ckpt=./data/model_weights/sector_extract.ckpt

#any input or output must be in project to ensure they get synced

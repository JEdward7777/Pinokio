#!/bin/bash

rl_zoo=/home/lansford/Sync/projects/tf_over/testing_stable_baselines/rl-baselines-zoo
pinokio=/home/lansford/Sync/projects/tf_over/pinokio/Pinokio
docker run -it --gpus all --rm --network host --ipc=host \
  --mount src=$rl_zoo,target=/root/code/rl_zoo,type=bind  \
  --mount src=$pinokio,target=/root/code/pinokio,type=bind \
  -h $(hostname) -e DISPLAY -e XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix -v $XAUTHORITY:$XAUTHORITY \
  stablebaselines/rl-baselines-zoo:v2.10.0\
  bash -c "cd /root/code/rl_zoo/ && bash"
  #bash -c "cd /root/code/rl_zoo/ && $cmd_line"

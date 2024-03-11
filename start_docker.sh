if [ "$1" = "" ]
then
  echo "Usage: $1 "
  exit
fi

hare run -it --gpus $1 -v /mnt/faster0/jrs68/wav2lip:/workspace -it --entrypoint "/bin/bash" --shm-size=256gb -t jrs68/wav2lip
mkdir -p ../data/lrs2
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partaa -O ../data/lrs2/lrs2_v1_partaa
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partab -O ../data/lrs2/lrs2_v1_partab
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partac -O ../data/lrs2/lrs2_v1_partac
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partad -O ../data/lrs2/lrs2_v1_partad
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partae -O ../data/lrs2/lrs2_v1_partae
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/pretrain.txt -O ../data/lrs2/pretrain.txt
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/train.txt -O ../data/lrs2/train.txt
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/val.txt -O ../data/lrs2/val.txt
wget --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/test.txt -O ../data/lrs2/test.txt

cd ../data/lrs2 && cat lrs2_v1_parta* > lrs2_v1.tar && tar -xvf lrs2_v1.tar


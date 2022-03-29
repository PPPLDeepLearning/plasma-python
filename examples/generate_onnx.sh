#!/usr/bin/bash -l

module load conda/2021-11-30; conda activate

length=(32)
rnn_size=(300)
rnn_layers=(6)

for l in ${length[*]}
do
    for size in ${rnn_size[*]}
    do
	for nlayer in ${rnn_layers[*]}
	do
	    cd /home/felker/plasma-python/examples
	    echo "STARTING bs_dynamic_layers${nlayer}_length${l}_size${size}.onnx"
	    sed -i "91 c\  length: ${l}" conf.yaml
	    sed -i "97 c\  rnn_size: ${size}" conf.yaml
	    sed -i "100 c\  rnn_layers: ${nlayer}" conf.yaml
	    python mpi_learn.py
	    mv /lus/theta-fs0/projects/fusiondl_aesp/felker/model_checkpoints/*.onnx ~/bs_dynamic_layers${nlayer}_length${l}_size${size}.onnx
	    rm -rfd /lus/theta-fs0/projects/fusiondl_aesp/felker/model_checkpoints/*
	    echo "FINISHED bs_dynamic_layers${nlayer}_length${l}_size${size}.onnx"
	done
    done
done

#!/bin/bash

source /home/rafa/.pyenv/versions/3.10.12/envs/iterated/bin/activate

export MODEL="/home/rafa/Desktop/roentgen"
0_img_to_vec.py \
    --pretrained_model_name_or_path="/home/rafa/Desktop/roentgen" \
    --data_dir="/home/rafa/Desktop/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/files" \
    --labels_csv="/home/rafa/Desktop/MIMIC/physionet.org/files/cxr-lt-iccv-workshop-cvamd/1.1.0/miccai-2023_mimic-cxr-lt/miccai2023_mimic-cxr-lt_labels_val.csv" \

python3 1_train_iterated.py \
    --pkl_file="data/latent_vecs.pkl" \
    --pretrained_model_name_or_path=$MODEL \
    --epochs=20 \

python3 2_fuse.py \
    --pkl_file="data/latent_vecs.pkl" \
    --model_path="/home/rafa/Desktop/il_se/iterated-model/classifier_weighted_0.pk" \
    --pretrained_model_name_or_path=$MODEL \
    --load=True \
    --d_steps=3 \
    
python3 3_vec_to_img.py \
    --pretrained_model_name_or_path="/home/rafa/Desktop/roentgen" \
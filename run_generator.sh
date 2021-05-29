#!/bin/bash
path_to_train="$1"
path_to_sample_imgs="$2"
generate9k="$3"
target9k="$4"

# SAVE ALL THE REQUIRED FILES IN ./part1_temp_saves/
mkdir ./part1_temp_saves # will give error if directory already exists

#load_data_from_query_target_images and save as numpy
python harman_scripts/load_sudoku_data.py --train_datapath "$1" --target_array_file "./part1_temp_saves/target_64k_images.npy" --query_array_file "./part1_temp_saves/query_64k_images.npy" --query_target_array_file "./part1_temp_saves/query_target_64k_images.npy"

#create supervised dataset using minibatch kmeans(for now)
python harman_scripts/clustering.py --savedir ./part1_temp_saves --query_datapath "./part1_temp_saves/query_64k_images.npy" --target_datapath "./part1_temp_saves/target_64k_images.npy" --oneshot_datapath "$2" --nclusters 9 --output_label_file "./part1_temp_saves/kmeans_mb_qt9c_labels.npy" --output_oneshot_label_file "./part1_temp_saves/kmeans_mb_qt9c_oneshot_labels.npy" --method minbatch-kmeans

#train a cgan
python harman_scripts/train_cgan.py --root_path_to_save "./part1_temp_saves/cgan_output" --traindatapath "./part1_temp_saves/query_target_64k_images.npy" --trainlabelspath  "./part1_temp_saves/kmeans_mb_qt9c_labels.npy" --train_or_gen train --num_epochs 150

#generate 9k images in form of npy files and save as gen9k.npy and target9k.npy
python harman_scripts/train_cgan.py --gen_model_pretr "./part1_temp_saves/cgan_output/gen_trained.pth" --gen9k_path "$3" --target9k_path "$4" --train_or_gen generate

#convert the generated npy images in png images,  and 9k real images to png images and save them and then calculate FID score
python harman_scripts/numpy2images.py --savedir "./part1_temp_saves/cgan_output/generated_images" --numpy_images_file "$3" --num_images 9000
python harman_scripts/numpy2images.py --savedir "./part1_temp_saves/real_images" --numpy_images_file "./part1_temp_saves/query_target_64k_images.npy" --num_images 9000
#calculate FID assuing we have gpu access
python -m pytorch_fid --device "cuda:0" "./part1_temp_saves/cgan_output/generated_images" "./part1_temp_saves/real_images"

# save the numpy files : generate9k and target9k
# print the FID score on terminal

#!/bin/bash

if [ $# -eq 4 ] # Part 2 
then
    path_to_train="$1"
    path_to_test_query="$2"
    path_to_sample_imgs="$3"
    path_to_out_csv="$4"
    # echo $1
    # echo $2
    # echo $3
    # echo $4
    # SAVE ALL THE REQUIRED FILES IN ./part2_temp_saves/
    mkdir ./part2_temp_saves 
    
    #load_data_from_query_target_images and save as numpy
    python harman_scripts/load_sudoku_data.py --train_datapath "$1" --target_array_file "./part2_temp_saves/target_64k_images.npy" --query_array_file "./part2_temp_saves/query_64k_images.npy" --query_target_array_file "./part2_temp_saves/query_target_64k_images.npy"

    #kmeans-minibatch-sampled to get 9k supervised data
    python harman_scripts/clustering.py --savedir "./part2_temp_saves/kmeans-sampled_15k_1000each" --query_datapath "./part2_temp_saves/query_64k_images.npy" --target_datapath "./part2_temp_saves/target_64k_images.npy" --oneshot_datapath "$3" --nclusters 9 --output_label_file "./part2_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --output_oneshot_label_file "./part2_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_oneshot_labels.npy" --method minibatch-kmeans-sampled --sampled_X_path "./part2_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy"

    #uda for making a classifier
    python harman_scripts/uda.py --testing_query_input "$2" --output_testing_query_labels "./part2_temp_saves/testing_query.npy" --output_classifier "./part2_temp_saves/uda_classifier.pth" --query_datapath "./part2_temp_saves/query_64k_images.npy" --target_datapath "./part2_temp_saves/target_64k_images.npy" --supervised_datapath "./part2_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy" --supervised_labels "./part2_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --output_qt_labels "./part2_temp_saves/uda_labels_qt.npy"

    #train rrn
    python part2_rrn_train.py --data_dir "./part2_temp_saves/uda_labels_qt.npy" --num_epochs 30 --num_steps 20 --exp_name part2_rrn --savemodel ./part2_temp_saves/ --saveplot ./part2_temp_saves/

    #test rrn
    python part2_rrn_test.py --data_dir "./part2_temp_saves/testing_query.npy" --num_steps 20 --model_path ./part2_temp_saves/part2_rrn.pth --output_csv $path_to_out_csv


    # # Do k-means clustering (and possibly training classifier using cGAN) using $1, $3 and save symbolic data as part2_train_symbolic_data.npy (size: 20k,8,8)
    # # Save symbolic data for $2 as well as part2_test_symbolic_data.npy (size: num_samples,8,8)


    # # rrn_train.py will load dataset from ./temp_saves (in numpy form: first 10k is target, rest 10k is query)
    # # it will store trained model in ./temp_saves 
    # python part2_rrn_train.py --data_dir ./part2_temp_saves/part2_train_symbolic_data.npy --num_epochs 30 --num_steps 20 --exp_name part2_rrn --savemodel ./part2_temp_saves/ --saveplot ./part2_temp_saves/
    # # rrn_test will load the same to do predictions and save in $4
    # # python part2_rrn_test.py --data_dir ./part2_temp_saves/part2_test_symbolic_data.npy --num_steps 20 --model_path ./part2_temp_saves/part2_rrn.pth --output_csv $path_to_out_csv
    # # [THIS TESTS ON THE TRAIN DATA]python part2_rrn_test.py --data_dir ./part2_temp_saves/part2_train_symbolic_data.npy --num_steps 20 --model_path ./part2_temp_saves/part2_rrn.pth --output_csv $path_to_out_csv

else # Part 3
    path_to_train="$1"
    path_to_test_query="$2"
    path_to_sample_imgs="$3"
    generate1k="$4"
    target1k="$5"
    path_to_out_csv="$6"

    # SAVE ALL THE REQUIRED FILES IN ./part3_temp_saves/
    mkdir ./part3_temp_saves 

    #load_data_from_query_target_images and save as numpy
    python harman_scripts/load_sudoku_data.py --train_datapath "$1" --target_array_file "./part3_temp_saves/target_64k_images.npy" --query_array_file "./part3_temp_saves/query_64k_images.npy" --query_target_array_file "./part3_temp_saves/query_target_64k_images.npy"

    #kmeans-minibatch-sampled to get 9k supervised data
    python harman_scripts/clustering.py --savedir "./part3_temp_saves/kmeans-sampled_15k_1000each" --query_datapath "./part3_temp_saves/query_64k_images.npy" --target_datapath "./part3_temp_saves/target_64k_images.npy" --oneshot_datapath "$3" --nclusters 9 --output_label_file "./part3_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --output_oneshot_label_file "./part3_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_oneshot_labels.npy" --method minibatch-kmeans-sampled --sampled_X_path "./part3_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy"

    #uda for making a classifier
    python harman_scripts/uda.py --testing_query_input "$2" --output_testing_query_labels "./part3_temp_saves/testing_query.npy" --output_classifier "./part3_temp_saves/uda_classifier.pth" --query_datapath "./part3_temp_saves/query_64k_images.npy" --target_datapath "./part3_temp_saves/target_64k_images.npy" --supervised_datapath "./part3_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy" --supervised_labels "./part3_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --output_qt_labels "./part3_temp_saves/uda_labels_qt.npy"

    #USE UDA CLASSIFIER TO JOINT TRAIN
    mkdir ./part3_temp_saves/saved_models
    mkdir ./part3_temp_saves/saved_results
    python harman_scripts/joint_train_algo6.py --epochs_wait_classif 5 --batch_size 128 --lreg_factor 0 --lr_classifier 5e-5 --lr_rrn 2e-3 --oneshot_file "./part3_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy" --oneshot_label_file "./part3_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --data_dir "./part3_temp_saves/query_target_64k_images.npy" --pretr_classifier "./part3_temp_saves/uda_classifier.pth" --loss_reg yes --num_epochs 50 --num_steps 20 --exp_name E_JOINT_TRAINING --savemodel "./part3_temp_saves/saved_models/" --saveplot "./part3_temp_saves/saved_results/"
    # python harman_scripts/joint_train_algo6.py --epochs_wait_classif 5 --batch_size 128 --lreg_factor 0 --lr_classifier 5e-5 --lr_rrn 2e-3 --oneshot_file "./part2_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy" --oneshot_label_file "./part2_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --data_dir "./part2_temp_saves/query_target_64k_images.npy" --pretr_classifier "./part2_temp_saves/uda_classifier.pth" --loss_reg yes --num_epochs 1 --num_steps 20 --exp_name E_JOINT_TRAINING --savemodel "./part3_temp_saves/saved_models" --saveplot "./part3_temp_saves/saved_results"


    #TEST THE RRN
    python harman_scripts/joint_test_algo6.py --data_dir_query_images "$2" --pretr_classifier "./part3_temp_saves/saved_models/E_JOINT_TRAINING_classifier.pth" --model_path "./part3_temp_saves/saved_models/E_JOINT_TRAINING_rrn.pth" --output_csv "$6"


    # use saved labels of uda classifier to train GAN
    python harman_scripts/train_cgan.py --root_path_to_save "./part3_temp_saves/cgan_output" --traindatapath "./part3_temp_saves/query_target_64k_images.npy" --trainlabelspath   "./part3_temp_saves/uda_labels_qt.npy" --train_or_gen train --num_epochs 100

    #generate 9k images in form of npy files and save as gen9k.npy and target9k.npy
    python harman_scripts/train_cgan.py --gen_model_pretr "./part3_temp_saves/cgan_output/gen_trained.pth" --gen9k_path "$5" --target9k_path "$6" --train_or_gen generate

    #convert the generated npy images in png images,  and 9k real images to png images and save them and then calculate FID score
    # python harman_scripts/numpy2images.py --savedir "./part3_temp_saves/cgan_output/generated_images" --numpy_images_file "$5" --num_images 9000
    # python harman_scripts/numpy2images.py --savedir "./part3_temp_saves/real_images" --numpy_images_file "./part3_temp_saves/query_target_64k_images.npy" --num_images 9000
    # #calculate FID assuing we have gpu access
    # python -m pytorch_fid --device "cuda:0" "./part3_temp_saves/cgan_output/generated_images" "./part3_temp_saves/real_images"


fi


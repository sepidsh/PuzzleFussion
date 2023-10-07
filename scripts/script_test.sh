MODEL_FLAGS="--dataset crosscut --batch_size 512 --set_name test"
#TRAIN_FLAGS="--lr 1e-3 --save_interval 5000 --weight_decay 0.05 --log_interval 500 --use_image_features False"
SAMPLE_FLAGS="--num_samples 1024  --dataset crosscut --batch_size 512  --set_name test --use_image_features False"

CUDA_VISIBLE_DEVICES='1' python image_sample.py  --set_name test --model_path ckpts/pred_s_wp_c_wimage/model130000.pt  $SAMPLE_FLAGS --exp_name pred_s_wp_c_wimage 


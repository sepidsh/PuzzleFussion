MODEL_FLAGS="--dataset voronoi --batch_size 1524 --set_name train"
TRAIN_FLAGS="--lr 1e-3 --save_interval 5000 --weight_decay 0.05 --log_interval 1000 --use_image_features False"
#SAMPLE_FLAGS="--batch_size 1024 --num_samples 32 --set_name test" 
mpiexec -n 2 python image_train.py $MODEL_FLAGS $TRAIN_FLAGS --exp_name preds

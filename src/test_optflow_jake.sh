#CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /data2/victorleee/grid/	 --name tellgan_ver1 --model tell_gan --continue_train

#python train_optflow_lstm.py --train --dataroot /home/jake/classes/cs703/Project/data/grid/	--features-model /home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat
python train_optflow_lstm.py --test --dataroot /home/jake/classes/cs703/Project/data/grid_test/ --ckptdir ./optflowGAN_chkpnts --features-model /home/jake/classes/cs703/Project/dev/TellGAN/src/assests/predictors/shape_predictor_68_face_landmarks.dat $@

@echo off
@REM RunMe.bat >> logs.txt


@REM Adapt the folder in the PATH to your system
@SET PATH=C:\ProgramData\Anaconda3\;C:\ProgramData\Anaconda3\Scripts;%PATH%

@REM conda activate tf_gpu
CALL activate.bat tf_gpu

@REM @REM -------------------------------------------------------------------------
@REM echo "Starting a new training."
@REM echo %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped3d_128_3 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --d_start_lr 1e-6 --d_lr_decay_factor 0.5 --mae_loss 1 --alpha 0 --beta 0 --mse_loss 0 --unrolling_iter 0

@REM @REM -------------------------------------------------------------------------
@REM echo "Starting a new training."
@REM echo %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped3d_128_3 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --d_start_lr 1e-6 --d_lr_decay_factor 0.5 --mae_loss 1 --alpha 0 --beta 0 --mse_loss 0 --unrolling_iter 0

@REM -------------------------------------------------------------------------
echo "Starting a new training."
echo %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped3d_128_3 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 1 --train_discriminator_times 0 --alpha 0 --beta 0 --gamma 0.2 --mae_loss 1 --mse_loss 0 --unrolling_iter 2

@echo off
@REM RunMe.bat >> logs.txt


@REM @REM Adapt the folder in the PATH to your system
@REM @SET PATH=C:\ProgramData\Anaconda3\;C:\ProgramData\Anaconda3\Scripts;%PATH%

@REM @REM conda activate tf_gpu
@REM CALL activate.bat tf_gpu

@REM @REM ------------------------------------------------------------------------- Done FixedCell_RCAN_14-04-2023_time0359
@REM echo "36. Starting a new training."
@REM echo %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 5 --n_RCAB 10 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --alpha 0 --beta 0 --mae_loss 1 --mse_loss 0 --unrolling_iter 0

@REM @REM ------------------------------------------------------------------------- FixedCell_RCAN_16-04-2023_time0340
@REM echo "[=============== *************** ===============]"
@REM echo "48. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --alpha 0 --beta 0 --mae_loss 1 --mse_loss 0 --unrolling_iter 0

@REM -------------------------------------------------------------------------
echo "[=============== *************** ===============]"
echo "49. Starting a new training."
echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --alpha 0 --beta 0.05 --mae_loss 1 --mse_loss 0 --unrolling_iter 0


@REM @REM ------------------------------------------------------------------------- Done FixedCell_RCAN_14-04-2023_time0438
@REM echo "37. Starting a new training."
@REM echo %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type RCAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 10 --epoch 500 --start_lr 1e-3 --lr_decay_factor 0.5 --alpha 0 --beta 0.5 --mae_loss 1 --mse_loss 0 --unrolling_iter 0

@REM -------------------------------------------------------------------------
@REM echo "[=============== *************** ===============]"
@REM echo "38. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 1 --train_discriminator_times 0 --alpha 0 --beta 0 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 2

@REM @REM ------------------------------------------------------------------------- FixedCell_UCAGAN_16-04-2023_time0420
@REM echo "[=============== *************** ===============]"
@REM echo "39. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 1 --train_discriminator_times 0 --alpha 0 --beta 0.5 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 2

@REM ------------------------------------------------------------------------- FixedCell_UCAGAN_16-04-2023_time0420
echo "[=============== *************** ===============]"
echo "51. Starting a new training."
echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 1 --train_discriminator_times 0 --alpha 0 --beta 0.05 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 2

@REM @REM ------------------------------------------------------------------------- FixedCell_CAGAN_16-04-2023_time0433
@REM echo "[=============== *************** ===============]"
@REM echo "40. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type CAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 5 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0 --beta 0 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 0

@REM @REM ------------------------------------------------------------------------- FixedCell_CAGAN_16-04-2023_time0528
@REM echo "[=============== *************** ===============]"
@REM echo "41. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type CAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 5 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0.1 --beta 0 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 0

@REM @REM ------------------------------------------------------------------------- FixedCell_CAGAN_16-04-2023_time0623
@REM echo "[=============== *************** ===============]"
@REM echo "42. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type CAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 5 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0 --beta 0.5 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 0

@REM @REM ------------------------------------------------------------------------- FixedCell_CAGAN_16-04-2023_time0719
@REM echo "[=============== *************** ===============]"
@REM echo "43. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type CAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 5 --n_RCAB 10 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0.1 --beta 0.5 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 0

@REM @REM ------------------------------------------------------------------------- FixedCell_UCAGAN_16-04-2023_time0815
@REM echo "[=============== *************** ===============]"
@REM echo "45. Starting a new training."
@REM echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

@REM python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0.1 --beta 0.5 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 2

@REM -------------------------------------------------------------------------
echo "[=============== *************** ===============]"
echo "50. Starting a new training."
echo "[=============== " %date:~-4% %date:~-10,2% %date:~-7,2% %time:~0,2% %time:~3,2% %time:~6,2%

python -m train --dnn_type UCAGAN --data_dir D:\Data\FixedCell\PFA_eGFP\cropped2d_128 --batch_size 8 --n_ResGroup 2 --n_RCAB 3 --n_channel 16 --epoch 500 --start_lr 1e-3 --d_start_lr 1e-6 --lr_decay_factor 0.5 --d_lr_decay_factor 0.5 --train_generator_times 3 --train_discriminator_times 1 --alpha 0.1 --beta 0.05 --gamma 0.2 --mae_loss 0 --mse_loss 1 --unrolling_iter 2

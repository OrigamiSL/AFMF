python -u main.py --model RTNet --data SMD --data_path ./data/SMD --input_len 720 --pyramid 3 --block_nums 3 --kernel 3 --dropout 0.1 --itr 5 --d_model 128 --batch_size 128 --variate 37 --out_variate 37 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model RTNet --data MSL --data_path ./data/MSL --input_len 48 --pyramid 3 --block_nums 3 --kernel 3 --dropout 0.1 --itr 5 --d_model 128 --batch_size 128 --variate 34 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model RTNet --data SMAP --data_path ./data/SMAP --input_len 24 --pyramid 1 --block_nums 1 --kernel 3 --dropout 0.1 --itr 5 --d_model 8 --batch_size 128 --variate 24 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model RTNet --data SWaT --data_path ./data/SWaT --input_len 720 --pyramid 3 --block_nums 3 --kernel 3 --dropout 0.1 --itr 5 --d_model 128 --batch_size 128 --variate 40 --out_variate 25 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model RTNet --data PSM --data_path ./data/PSM --input_len 720 --pyramid 3 --block_nums 3 --kernel 3 --dropout 0.1 --itr 5 --d_model 128 --batch_size 128 --variate 25 --out_variate 25 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model DLinear --data SMD --data_path ./data/SMD --input_len 720 --kernel 25 --dropout 0.1 --itr 5 --batch_size 128 --variate 37 --learning_rate 0.0001 --out_variate 37 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model DLinear --data MSL --data_path ./data/MSL --input_len 48 --kernel 25 --dropout 0.1 --itr 5 --batch_size 128 --variate 34 --learning_rate 0.0001 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model DLinear --data SMAP --data_path ./data/SMAP --input_len 24 --kernel 11 --dropout 0.1 --itr 5 --batch_size 128 --variate 24 --learning_rate 0.0001 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model DLinear --data SWaT --data_path ./data/SWaT --input_len 720 --kernel 25 --dropout 0.1 --itr 5 --batch_size 128 --variate 40 --learning_rate 0.0001 --out_variate 25 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model DLinear --data PSM --data_path ./data/PSM --input_len 720 --kernel 25 --dropout 0.1 --itr 5 --batch_size 128 --variate 25 --learning_rate 0.0001 --out_variate 25 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

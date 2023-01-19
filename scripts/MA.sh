python -u main.py --model MA --data SMD --data_path ./data/SMD --input_len 720 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 37 --out_variate 37 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data MSL --data_path ./data/MSL --input_len 48 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 34 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data SMAP --data_path ./data/SMAP --input_len 24 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 24 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data SWaT --data_path ./data/SWaT --input_len 720 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 40 --out_variate 25 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data PSM --data_path ./data/PSM --input_len 720 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 25 --out_variate 25 --anomaly_ratio 1 --retrain --detection_adjustment --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data WADI --data_path ./data/WADI --input_len 100 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 92 --out_variate 66 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 0 --thresh 3 --data_process --LIN

python -u main.py --model MA --data MBA --data_path ./data/MBA --input_len 100 --itr 1 --learning_rate 0.0001 --batch_size 128 --variate 2 --out_variate 2 --anomaly_ratio 2 --detection_adjustment --retrain --drop 0 --thresh 3 --data_process --LIN

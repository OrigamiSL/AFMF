python -u main.py --model GTA --data SMD --data_path ./data/SMD --input_len 720 --label_len 30 --variate 37 --out_variate 37 --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data MSL --data_path ./data/MSL --input_len 48 --label_len 30 --variate 34 --out_variate 1 --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data SMAP --data_path ./data/SMAP --input_len 24 --label_len 12 --variate 24 --out_variate 1  --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data SWaT --data_path ./data/SWaT --input_len 720 --label_len 30 --variate 40 --out_variate 25 --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data PSM --data_path ./data/PSM --input_len 720 --label_len 30 --variate 25 --out_variate 25  --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data WADI --data_path ./data/WADI --input_len 100 --label_len 30 --variate 92 --out_variate 66  --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GTA --data MBA --data_path ./data/MBA --input_len 100 --label_len 30 --variate 2 --out_variate 2  --d_model 512 --e_layers 2 --d_layers 1  --factor 5 --dropout 0.1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 2 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

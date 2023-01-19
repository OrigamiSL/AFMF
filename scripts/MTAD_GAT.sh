python -u main.py --model MTAD_GAT --data SMD --data_path ./data/SMD --input_len 720 --gru_n_layers 1 --gru_hid_dim 150 --fc_n_layers 3 --fc_hid_dim 150 --alpha 0.2 --variate 37 --out_variate 37 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model MTAD_GAT --data MSL --data_path ./data/MSL --input_len 48 --gru_n_layers 1 --gru_hid_dim 150 --fc_n_layers 3 --fc_hid_dim 150 --alpha 0.2 --variate 34 --out_variate 1 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model MTAD_GAT --data SMAP --data_path ./data/SMAP --input_len 24 --gru_n_layers 1 --gru_hid_dim 150 --fc_n_layers 3 --fc_hid_dim 150 --alpha 0.2 --variate 24 --out_variate 1  --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model MTAD_GAT --data SWaT --data_path ./data/SWaT --input_len 720 --gru_n_layers 1 --gru_hid_dim 150 --fc_n_layers 3 --fc_hid_dim 150 --alpha 0.2 --variate 40 --out_variate 25 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model MTAD_GAT --data PSM --data_path ./data/PSM --input_len 720 --gru_n_layers 1 --gru_hid_dim 150 --fc_n_layers 3 --fc_hid_dim 150 --alpha 0.2 --variate 25 --out_variate 25 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

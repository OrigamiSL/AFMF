python -u main.py --model GDN --data SMD --data_path ./data/SMD --input_len 720 --variate 37 --out_variate 37 --out_layer_num 1 --out_layer_inter_dim 256 --topk 20  --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GDN --data MSL --data_path ./data/MSL --input_len 48 --variate 34 --out_variate 1 --out_layer_num 1 --out_layer_inter_dim 256 --topk 20 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GDN --data SMAP --data_path ./data/SMAP --input_len 24 --variate 24 --out_variate 1  --out_layer_num 1 --out_layer_inter_dim 256 --topk 20 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GDN --data SWaT --data_path ./data/SWaT --input_len 720 --variate 40 --out_variate 25 --out_layer_num 1 --out_layer_inter_dim 256 --topk 20 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 0.5 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model GDN --data PSM --data_path ./data/PSM --input_len 720 --variate 25 --out_variate 25  --out_layer_num 1 --out_layer_inter_dim 256 --topk 20 --itr 5 --learning_rate 0.0001 --batch_size 128 --anomaly_ratio 1 --detection_adjustment --retrain --drop 4 --thresh 3 --data_process --LIN

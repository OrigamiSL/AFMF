python -u main.py --model Informer --data SMD --data_path ./data/SMD --input_len 720 --label_len 360 --e_layers 2 --d_layers 1 --factor 5 --attn prob --dropout 0.1 --itr 5 --d_model 512 --d_ff 2048 --distil --mix --batch_size 128 --variate 37 --out_variate 37 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model Informer --data MSL --data_path ./data/MSL --input_len 48 --label_len 24 --e_layers 2 --d_layers 1 --factor 5 --attn prob --dropout 0.1 --itr 5 --d_model 512 --d_ff 2048 --distil --mix --batch_size 128 --variate 34 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model Informer --data SMAP --data_path ./data/SMAP --input_len 24 --label_len 12 --e_layers 2 --d_layers 1 --factor 5 --attn prob --dropout 0.1 --itr 5 --d_model 512 --d_ff 2048 --distil --mix --batch_size 128 --variate 24 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model Informer --data SWaT --data_path ./data/SWaT --input_len 720 --label_len 360 --e_layers 2 --d_layers 1 --factor 5 --attn prob --dropout 0.1 --itr 5 --d_model 512 --d_ff 2048 --distil --mix --batch_size 128 --variate 40 --out_variate 25 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model Informer --data PSM --data_path ./data/PSM --input_len 720 --label_len 360 --e_layers 2 --d_layers 1 --factor 5 --attn prob --dropout 0.1 --itr 5 --d_model 512 --d_ff 2048 --distil --mix --batch_size 128 --variate 25 --out_variate 25 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

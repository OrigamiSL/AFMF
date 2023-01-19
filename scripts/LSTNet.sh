python -u main.py --model LSTNet --data SMD --data_path ./data/SMD --input_len 720 --RNN_hid_size 512 --CNN_hid_size 100 --CNN_kernel 6 --highway_window 0 --skip 24 --hidSkip 5 --dropout 0.1 --itr 5 --batch_size 128 --variate 37 --out_variate 37 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model LSTNet --data MSL --data_path ./data/MSL --input_len 48 --RNN_hid_size 512 --CNN_hid_size 100 --CNN_kernel 6 --highway_window 0 --skip 24 --hidSkip 5 --dropout 0.1 --itr 5 --batch_size 128 --variate 34 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model LSTNet --data SMAP --data_path ./data/SMAP --input_len 24 --RNN_hid_size 512 --CNN_hid_size 100 --CNN_kernel 6 --highway_window 0 --skip 12 --hidSkip 5 --dropout 0.1 --itr 5 --batch_size 128 --variate 24 --out_variate 1 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model LSTNet --data SWaT --data_path ./data/SWaT --input_len 720 --RNN_hid_size 512 --CNN_hid_size 100 --CNN_kernel 6 --highway_window 0 --skip 24 --hidSkip 5 --dropout 0.1 --itr 5 --batch_size 128 --variate 40 --out_variate 25 --anomaly_ratio 0.5 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

python -u main.py --model LSTNet --data PSM --data_path ./data/PSM --input_len 720 --RNN_hid_size 512 --CNN_hid_size 100 --CNN_kernel 6 --highway_window 0 --skip 24 --hidSkip 5 --dropout 0.1 --itr 5 --batch_size 128 --variate 25 --out_variate 25 --anomaly_ratio 1 --retrain --detection_adjustment --drop 4 --thresh 3 --data_process --LIN

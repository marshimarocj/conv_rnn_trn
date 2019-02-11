Usage Documentation
====
```sh
usage: conv_rnn.py [-h] [--is_train IS_TRAIN] [--best_epoch BEST_EPOCH]
                   [--data_dir DATA_DIR] [--out_name OUT_NAME]
                   model_cfg_file path_cfg_file

functions: two modes, trn mode for trn and validation, tst mode for tst (prediction)
    

positional arguments:
  model_cfg_file        
                         configuration file of model and train paramters
                            
  path_cfg_file         
                        configuration file of data and experiment directories
                            

optional arguments:
  -h, --help            show this help message and exit
  --is_train IS_TRAIN   
                        1 for train mode and 0 for test mode
                            
  --best_epoch BEST_EPOCH
                        
                        option only in tst mode, the epoch used in tst
                            
  --data_dir DATA_DIR   
                        option only in tst mode, the data_dir used in tst
                            
  --out_name OUT_NAME   
                        option only in tst mode, the output file name in tst
```

Technical Details
====
1. look at /app/trntst/unit_tst/prepare_data.py for the code to prepare data for trn and tst, the data is stored in tf records format
2. look at /app/trntst/unit_tst/prepare_cfg.py for the code to generate model_cfg_file and path_cfg_file
3. look at /app/trntst/unit_tst/trn.sh for the script to run trn mode
4. look at /app/trntst/unit_tst/tst.sh for the script to run tst mode

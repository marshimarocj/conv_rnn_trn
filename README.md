Usage Documentation
====
```sh
root@ec833b506ef3:/home/jiac/code/python/template/diva_conv_rnn_trn/trntst/driver# python conv_rnn.py -h
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
====
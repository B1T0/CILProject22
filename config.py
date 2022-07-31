import json 

"""
Training and evaluation settings
"""
config = dict()

"""
Training or inference mode
"""
config['train_path'] = 'data/raw/train_split_'  # 'train' or 'eval'
config['val_path'] = 'data/raw/test_split_'

config['start_split'] = 0
config['num_splits'] = 5
config['dataset'] = 'data/raw/data_train.csv'

config['predict_dir'] = 'reports/logs/20220731-212306_graph_auto_encoder_32'
"""
Model related settings 
"""
config['model'] = 'GraphAttention'

"""
Training related settings
"""
# Most of them are moved to hyperparameters.py for model specific settings

"""
Logging and Analysis 
"""
config['results_dir'] = 'reports/logs'

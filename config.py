import json 

"""
Training and evaluation settings
"""
config = dict()

"""
Training or inference mode
"""
config['mode'] = 'train'  # 'train' or 'eval'

"""
Data related settings 
"""
config['dataset'] = 'FIRST'  # options: sleep-edf-153, ...
# load input size from json file of the dataset 
with open(f"data/processed/{config['dataset']}/info.json") as f:
    data = json.load(f)
    config['input_width'] =  data['input_width'] 
    config['input_height'] = data['input_height']

"""
Model related settings 
Available models: Acceptor
"""
config['model'] = 'Acceptor'

"""
Training related settings
"""
# Most of them are moved to hyperparameters.py for model specific settings

"""
Logging and Analysis 
"""
config['results_dir'] = 'reports/logs'

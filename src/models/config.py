"""
In this file we configure which model to run and what to to with it
We also select the dataset which we want to operate on
"""
config = dict()

"""
Select which model to run 

Available models: DETRtime, ... 
Available modes: 'train', 'test', 'inference' 
"""
config['model'] = 'DETRtime'
config['mode'] = 'train'


"""
Dataset related settings 
"""
config['dataset'] = 'EEG'
config['batch_size'] = 32
config['shuffle'] = True

"""
Training related settings
"""
config['gpu_ids'] = [0]  # only use one gpu

"""
Evaluation related settings 
"""
#TODO:implement
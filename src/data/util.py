from src.data.RowDataset.graph_datamodule import RowDataset
from src.data.GraphAutoencoder.graph_datamodule import Graph_Dataset


def get_dataset(model_name: str, file_path:str , args: dict):
    if model_name == 'GraphAttention':
        return RowDataset(file_path=file_path, n_users=10000, n_items=1000, user=args['mode'])
    elif model_name == 'GraphUserEncoder':
        return RowDataset(file_path=file_path, n_users=10000, n_items=1000, user=args['mode'])
    elif model_name == 'GraphAutoencoder':
        return Graph_Dataset(file_path=file_path, n_items=1000, n_users=10000)
    elif model_name == 'SVDGraphAttention':
        return Graph_Dataset(file_path=file_path, n_items=1000, n_users=10000)
    else:
        raise NotImplementedError(f'Dataloader for  {model_name} is not implemented')

from src.models.hyperparameters import params
from src.models.GraphAttention.model import GraphAttention
from src.models.GraphAutoencoder.model import GraphAutoencoder
from src.models.GraphUserEncoder.model import GraphUserEncoder
from src.models.SVDGraphAttention.model import SVDGraphAttention

from src.models.GraphAttention.predict_graph_attention import predict_graph_attention
from src.models.GraphUserEncoder.predict_graph_user_encoder import predict_graph_user_encoder
from src.models.GraphAutoencoder.predict_graphautoencoder import predict_graph_autoencoder
from src.models.SVDGraphAttention.predict import predict_svd_graph_attention

def get_model_params(model_name: str):
    if model_name == 'GraphAttention':
        return params[model_name]
    elif model_name == 'GraphUserEncoder':
        return params[model_name]
    elif model_name == 'GraphAutoencoder':
        return params[model_name]
    elif model_name == 'SVDGraphAttention':
        return params[model_name]
    else:
        raise NotImplementedError('Model is not implemented')


def get_model(model_name: str, hyper_params: dict, file_name: str):
    if model_name == 'GraphAttention':
        return GraphAttention(file_path=file_name, **hyper_params)
    elif model_name == 'GraphAutoencoder':
        return GraphAutoencoder(file_path=file_name, **hyper_params)
    elif model_name == 'GraphUserEncoder':
        return GraphUserEncoder(file_path=file_name, **hyper_params)
    elif model_name == 'SVDGraphAttention':
        return SVDGraphAttention(file_path=file_name, **hyper_params)
    else:
        raise NotImplementedError('Model is not implemented')


def predict_model(log_dir, model_name, model, args, dataloader, split=0):
    if model_name == 'GraphAttention':
        return predict_graph_attention(log_dir, model, args, dataloader, split=split)
    elif model_name == 'GraphUserEncoder':
        return predict_graph_user_encoder(log_dir, model, args, dataloader, split=split)
    elif model_name == 'GraphAutoencoder':
        return predict_graph_autoencoder(log_dir, model, args, dataloader, split=split)
    elif model_name == 'SVDGraphAttention':
        return predict_svd_graph_attention(log_dir, model, args, dataloader, split=split)

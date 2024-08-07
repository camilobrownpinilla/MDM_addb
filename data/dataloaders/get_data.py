from torch.utils.data import DataLoader
from data.dataloaders.tensors import collate 
from utils import config
import os

def get_dataset_class(name):
    if name == 'addb':
        from data.AddBiomechanicsDataset import AddBiomechanicsDataset
        return AddBiomechanicsDataset
    else:
        raise ValueError(f'Unsupported dataset name: {name}')
    
def get_dataset(name, window_size, split):
    if split not in ['train', 'test']:
        raise ValueError('Unsupported Data Split')
    DATA = get_dataset_class(name)
    data_path = os.path.join(config.DATA_PATH, split)
    geometry_folder = config.GEOMETRY_PATH
    dataset = DATA(window_size=window_size, data_path=data_path, geometry_folder=geometry_folder, skip_loading_skeletons=True)
    return dataset

def get_dataset_loader(name, batch_size, window_size, split):
    dataset = get_dataset(name, window_size, split)
    collate_fn = collate

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, drop_last=True
    )

    return loader
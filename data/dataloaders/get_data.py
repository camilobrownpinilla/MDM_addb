from torch.utils.data import DataLoader
from dataloaders.tensors import collate 

def get_dataset_class(name):
    if name == 'addb':
        from data.AddBiomechanicsDataset import AddBiomechanicsDataset
        return AddBiomechanicsDataset
    else:
        raise ValueError(f'Unsupported dataset name: {name}')
    
def get_dataset(name, window_size):
    DATA = get_dataset_class(name)
    dataset = DATA(window_size=window_size)
    return dataset

def get_dataset_loader(name, batch_size, window_size):
    dataset = get_dataset(name, window_size)
    collate = collate

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader
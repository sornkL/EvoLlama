import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    def __init__(self, dataset_path: str, pdb_dir: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.pdb_dir = pdb_dir
        if self.pdb_dir[-1] != '/':
            self.pdb_dir += '/'

        if dataset_path.endswith('json'):
            self.data = pd.read_json(dataset_path)
        elif dataset_path.endswith('jsonl'):
            self.data = pd.read_json(dataset_path, lines=True)
        else:
            raise ValueError('MultimodalDataset only supports json and jsonl file formats.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'pdb_files': None if pd.isna(self.data.iloc[idx]['structure']).any() else [self.pdb_dir + struct for struct in self.data.iloc[idx]['structure']],
            'sequences': self.data.iloc[idx]['sequence'],
            'prompts': self.data.iloc[idx]['conversations'][0]['value'],  # single-turn conversation
            'responses': self.data.iloc[idx]['conversations'][1]['value']  # single-turn conversation
        }


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        batch_dict[key] = [sample[key] for sample in batch]
    return batch_dict

import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_length, feature_dim):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.randint(1, 200, (self.seq_length,))
        sample1 = torch.randint(1, 8, (self.seq_length,))
        sample2 = torch.randint(1, 8, (self.seq_length,))
        sample3 = torch.randint(1, 9, (self.seq_length,))
        return sample,  sample1,  sample2,  sample3

batch_size = 16
num_samples = 1000  
seq_length = 128
feature_dim = 768

random_dataset = RandomDataset(num_samples=num_samples, seq_length=seq_length, feature_dim=feature_dim)

random_dataloader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)
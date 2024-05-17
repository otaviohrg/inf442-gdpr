import os
import pandas as pd
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  
  train_data = pd.read_csv(train_dir)
  test_data = pd.read_csv(test_dir)

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
  )

  return train_dataloader, test_dataloader
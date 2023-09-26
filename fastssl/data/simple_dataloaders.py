import numpy as np
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self,
                 data_labels_dict,
                 x_key='activations',
                 y_key='labels',
                 gt_key=None,
                 sid_key=None):
        self.x = data_labels_dict[x_key]
        self.y = data_labels_dict[y_key]
        self.gt = None
        self.sid = None
        if gt_key is not None:
            self.gt = data_labels_dict[gt_key] # ground_truth
        if sid_key is not None:
            self.sid = data_labels_dict[sid_key] # sample_id

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        retval = (self.x[idx], self.y[idx])
        if self.gt is not None:
            retval += (self.gt[idx],)
        if self.sid is not None:
            retval += (self.sid[idx],)
        return retval

def simple_dataloader(fname_train,
                     fname_test,
                     splits=['train','test'],
                     batch_size=512,
                     num_workers=2,
                     label_noise=0):
    assert fname_train==fname_test, "Precaching train/test features should be stored in the same file!"
    data_from_file = np.load(fname_train,allow_pickle=True).item()
    loaders = {}
    for split in splits:
        if label_noise > 0 and split == 'train':
            dataset = SimpleDataset(data_labels_dict=data_from_file[split], gt_key='ground_truths', sid_key='sample_ids')
        else:
            dataset = SimpleDataset(data_labels_dict=data_from_file[split])
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if split=='train' else False,
            num_workers=num_workers
        )

    return loaders

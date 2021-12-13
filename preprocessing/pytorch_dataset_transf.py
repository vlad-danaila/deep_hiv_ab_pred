import torch as t
from deep_hiv_ab_pred.util.tools import to_torch

class AssayDataset(t.utils.data.Dataset):

    def __init__(self, assays, antibody_seq, virus_seq):
        super().__init__()
        self.assays = assays
        self.antibody_seq = antibody_seq
        self.virus_seq = virus_seq

    def __getitem__(self, i):
        id, antibody, virus, ground_truth = self.assays[i]
        ab_tensor = self.antibody_seq[antibody]
        virus_tensor, pngs_mask_tensor = self.virus_seq[virus]

        ab_tensor = to_torch(ab_tensor, type=t.int)
        virus_tensor = to_torch(virus_tensor, type=t.int)
        pngs_mask_tensor = to_torch(pngs_mask_tensor, type=t.float32)
        ground_truth = to_torch(ground_truth, type=t.float32)

        return ab_tensor, virus_tensor, pngs_mask_tensor, ground_truth

    def __len__(self):
        return len(self.assays)
import torch as t
from deep_hiv_ab_pred.util.tools import device
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
from deep_hiv_ab_pred.util.tools import to_torch

class AssayDataset(t.utils.data.Dataset):

    def __init__(self, assays, antibody_cdrs, virus_seq, pngs_mask_to_kemr_tensor):
        super().__init__()
        self.assays = assays
        self.antibody_cdrs = antibody_cdrs
        self.virus_seq = virus_seq
        self.pngs_mask_to_kemr_tensor = pngs_mask_to_kemr_tensor

    def __getitem__(self, i):
        id, antibody, virus, ground_truth = self.assays[i]
        ab_cdr_tensor, ab_cdr_position_tensor = self.antibody_cdrs[antibody]
        virus_tensor = self.virus_seq[virus]
        pngs_mask_tensor = self.pngs_mask_to_kemr_tensor[virus]
        return ab_cdr_tensor, ab_cdr_position_tensor, virus_tensor, pngs_mask_tensor, ground_truth

    def __len__(self):
        return len(self.assays)

def __len__(self):
    return len(self.assays)

def zero_padding(batch):
    ab_cdr     = to_torch([b[0] for b in batch], type = t.int)
    ab_cdr_pos = to_torch([b[1] for b in batch])
    virus        = [t.tensor(b[2], dtype=t.int, device = device) for b in batch]
    pngs_mask    = [t.tensor(b[3], dtype=t.float32, device = device) for b in batch]
    batched_ground_truth = t.tensor([b[4] for b in batch], dtype=t.float32, device=device)
    batched_virus = t.nn.utils.rnn.pad_sequence(virus, batch_first=True, padding_value = amino_to_index['X'])
    batched_pngs_mask = t.nn.utils.rnn.pad_sequence(pngs_mask, batch_first=True, padding_value = 0)
    return ab_cdr, ab_cdr_pos, batched_virus, batched_pngs_mask, batched_ground_truth
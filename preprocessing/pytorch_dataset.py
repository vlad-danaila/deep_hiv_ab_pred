import torch as t
from deep_hiv_ab_pred.util.tools import device
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index

class AssayDataset(t.utils.data.Dataset):

    def __init__(self, assays, antibody_light_seq, antibody_heavy_seq, virus_seq, pngs_mask_to_kemr_tensor):
        super().__init__()
        self.assays = assays
        self.antibody_light_seq = antibody_light_seq
        self.antibody_heavy_seq = antibody_heavy_seq
        self.virus_seq = virus_seq
        self.pngs_mask_to_kemr_tensor = pngs_mask_to_kemr_tensor

    def __getitem__(self, i):
        id, antibody, virus, ground_truth = self.assays[i]
        antibody_light_tensor = self.antibody_light_seq[antibody]
        antibody_heavy_tensor = self.antibody_heavy_seq[antibody]
        virus_tensor          = self.virus_seq[virus]
        pngs_mask_tensor      = self.pngs_mask_to_kemr_tensor[virus]
        return antibody_light_tensor, antibody_heavy_tensor, virus_tensor, pngs_mask_tensor, ground_truth

    def __len__(self):
        return len(self.assays)

def __len__(self):
    return len(self.assays)

def zero_padding(batch):
    ab_light     = [t.nn.functional.pad(b[0], (0, LIGHT_ANTIBODY_TRIM - len(b[0])), value = amino_to_index['-']) for b in batch]
    ab_heavy     = [t.nn.functional.pad(b[1], (0, HEAVY_ANTIBODY_TRIM - len(b[1])), value = amino_to_index['-']) for b in batch]
    virus        = [b[2] for b in batch]
    pngs_mask    = [b[3] for b in batch]
    ground_truth = [b[4] for b in batch]
    batched_ab_light = t.stack(ab_light)
    batched_ab_heavy = t.stack(ab_heavy)
    batched_virus = t.nn.utils.rnn.pad_sequence(virus, batch_first=True, padding_value=0)
    #batched_virus = t.stack(virus)
    batched_pngs_mask = t.nn.utils.rnn.pad_sequence(pngs_mask, batch_first=True, padding_value=0)
    #batched_pngs_mask = t.stack(pngs_mask)
    batched_ground_truth = t.tensor(ground_truth, dtype=t.float32, device=device)
    # print(batched_ab_light.shape, batched_ab_heavy.shape, batched_virus.shape, batched_ground_truth.shape)
    return batched_ab_light, batched_ab_heavy, batched_virus, batched_pngs_mask, batched_ground_truth
from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.constants import CDR_LENGHTS
from deep_hiv_ab_pred.global_constants import INCLUDE_CDR_MASK_FEATURES, INCLUDE_CDR_POSITION_FEATURES
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix


class Ab_ATT_FC(t.nn.Module):

    def __init__(self, in_size, out_size, conf):
        super().__init__()
        self.att = t.nn.Linear(in_size, in_size)
        self.fc = t.nn.Linear(in_size, out_size)

        t.nn.TransformerEncoderLayer
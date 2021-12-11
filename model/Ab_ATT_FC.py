from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.constants import CDR_LENGHTS
from deep_hiv_ab_pred.global_constants import INCLUDE_CDR_MASK_FEATURES, INCLUDE_CDR_POSITION_FEATURES
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix


class Ab_ATT_FC(t.nn.Module):

    def __init__(self, in_size, out_size, conf, include_norm):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.conf = conf
        self.include_norm = include_norm

        self.att = t.nn.Linear(in_size, in_size)
        self.fc = t.nn.Linear(in_size, out_size)

        if self.include_norm:
            self.norm_1 = t.nn.LayerNorm(in_size)
            self.norm_2 = t.nn.LayerNorm(out_size)

        self.dropout_1 = t.nn.Dropout(conf['ANTIBODIES_DROPOUT_ATT'])
        self.dropout_2 = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])

    def forward(self, cdr_tensor):
        pass
        
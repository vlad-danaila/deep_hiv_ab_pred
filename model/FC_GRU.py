from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.global_constants import INCLUDE_CDR_POSITION_FEATURES
from deep_hiv_ab_pred.preprocessing.constants import AB_CDRS_SEQ_LEN, AB_CDRS_POS_LEN
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix

class FC_GRU(t.nn.Module):

    def __init__(self, conf, embeddings_matrix = None, include_position_features = False):
        super().__init__()
        self.conf = conf
        self.include_position_features = include_position_features
        if embeddings_matrix is None:
            self.embeding_size = conf['EMBEDDING_SIZE']
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
        else:
            self.embeding_size =  embeddings_matrix.shape[1]
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
            self.aminoacid_embedding.load_state_dict({'weight': embeddings_matrix})
            self.aminoacid_embedding.weight.requires_grad = False
        AB_TENSOR_SIZE = AB_CDRS_SEQ_LEN * self.embeding_size + (AB_CDRS_POS_LEN if include_position_features else 0)
        self.ab_fc = t.nn.Linear(AB_TENSOR_SIZE, conf['RNN_HIDDEN_SIZE'])
        self.ab_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.VIRUS_RNN_HIDDEN_SIZE = conf['RNN_HIDDEN_SIZE']
        self.virus_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_VIRUS'] * self.embeding_size + conf['KMER_LEN_VIRUS'],
            hidden_size = self.VIRUS_RNN_HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )
        self.embedding_dropout = t.nn.Dropout(conf['EMBEDDING_DROPOUT'])
        self.fc_dropout = t.nn.Dropout(conf['FULLY_CONNECTED_DROPOUT'])
        self.fully_connected = t.nn.Linear(2 * self.VIRUS_RNN_HIDDEN_SIZE, 1)
        self.sigmoid = t.nn.Sigmoid()

    def virus_state_init(self, batch_size):
        return t.zeros(2, batch_size, self.VIRUS_RNN_HIDDEN_SIZE, device=device)

    def forward_embeddings(self, ab_cdr, virus, batch_size):
        ab_cdr_seq = self.aminoacid_embedding(ab_cdr).reshape(batch_size, -1)
        virus = self.aminoacid_embedding(virus).reshape(batch_size, -1, self.conf['KMER_LEN_VIRUS'] * self.embeding_size)
        ab_cdr_seq = self.embedding_dropout(ab_cdr_seq)
        virus = self.embedding_dropout(virus)
        return ab_cdr_seq, virus

    def forward_antibodyes(self, ab_cdr, ab_cdr_pos = None):
        if ab_cdr_pos:
            ab = t.cat([ab_cdr, ab_cdr_pos], axis = 1)
            return self.ab_dropout(self.ab_fc(ab))
        else:
            return self.ab_dropout(self.ab_fc(ab_cdr))

    def forward_virus(self, virus, pngs_mask, ab_hidden):
        virus_and_pngs = t.cat([virus, pngs_mask], axis = 2)
        self.virus_gru.flatten_parameters()
        ab_hidden_clone = t.clone(ab_hidden)
        ab_hidden_bidirectional = t.stack((ab_hidden, ab_hidden_clone), axis = 0)
        virus_ab_all_output, _ = self.virus_gru(virus_and_pngs, ab_hidden_bidirectional)
        virus_output = virus_ab_all_output[:, -1]
        virus_output = self.fc_dropout(virus_output)
        return self.sigmoid(self.fully_connected(virus_output).squeeze())

    def forward(self, ab_cdr, ab_cdr_pos, virus, pngs_mask):
        batch_size = len(ab_cdr)
        ab_cdr, virus = self.forward_embeddings(ab_cdr, virus, batch_size)
        if self.include_position_features and ab_cdr_pos:
            ab_hidden = self.forward_antibodyes(ab_cdr, ab_cdr_pos)
        else:
            ab_hidden = self.forward_antibodyes(ab_cdr)
        return self.forward_virus(virus, pngs_mask, ab_hidden)

def get_FC_GRU_model(conf):
    model = FC_GRU(conf, get_embeding_matrix(), INCLUDE_CDR_POSITION_FEATURES).to(device)
    model = t.nn.DataParallel(model)
    return model
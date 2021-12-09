from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix
from deep_hiv_ab_pred.util.tools import device

class FC_TRANS_GRU(t.nn.Module):

    def __init__(self, conf, embeddings_matrix = None):
        super().__init__()
        self.conf = conf
        if embeddings_matrix is None:
            self.embeding_size = conf['EMBEDDING_SIZE']
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
        else:
            self.embeding_size =  embeddings_matrix.shape[1]
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
            self.aminoacid_embedding.load_state_dict({'weight': embeddings_matrix})
            self.aminoacid_embedding.weight.requires_grad = False
        self.light_ab_trans = t.nn.TransformerEncoderLayer(self.embeding_size * LIGHT_ANTIBODY_TRIM, conf['AB_TRANS_HEADS'],
            conf['AB_TRANS_FC'], conf['AB_TRANS_DROPOUT'], 'relu', conf['AB_TRANS_NORM'], True, device, t.float32)
        self.light_ab_fc = t.nn.Linear(LIGHT_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.light_ab_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.heavy_ab_fc = t.nn.Linear(HEAVY_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.heavy_ab_trans = t.nn.TransformerEncoderLayer(self.embeding_size * HEAVY_ANTIBODY_TRIM, conf['AB_TRANS_HEADS'],
            conf['AB_TRANS_FC'], conf['AB_TRANS_DROPOUT'], 'relu', conf['AB_TRANS_NORM'], True, device, t.float32)
        self.heavy_ab_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.VIRUS_RNN_HIDDEN_SIZE = conf['RNN_HIDDEN_SIZE'] * 2
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

    def forward_embeddings(self, ab_light, ab_heavy, virus, batch_size):
        ab_light = self.aminoacid_embedding(ab_light).reshape(batch_size, -1)
        ab_heavy = self.aminoacid_embedding(ab_heavy).reshape(batch_size, -1)
        virus = self.aminoacid_embedding(virus).reshape(batch_size, -1, self.conf['KMER_LEN_VIRUS'] * self.embeding_size)
        ab_light = self.embedding_dropout(ab_light)
        ab_heavy = self.embedding_dropout(ab_heavy)
        virus = self.embedding_dropout(virus)
        return ab_light, ab_heavy, virus

    def forward_antibodyes(self, ab_light, ab_heavy, batch_size):
        ab_light = ab_light.reshape((batch_size, 1, -1))
        ab_heavy = ab_heavy.reshape((batch_size, 1, -1))
        light_ab_trans = self.light_ab_trans(ab_light)
        heavy_ab_trans = self.heavy_ab_trans(ab_heavy)
        light_ab_trans = light_ab_trans.reshape((batch_size, -1))
        heavy_ab_trans = heavy_ab_trans.reshape((batch_size, -1))
        # self.light_ab_fc.flatten_parameters()
        light_ab_hidden = self.light_ab_fc(light_ab_trans)
        # self.heavy_ab_fc.flatten_parameters()
        heavy_ab_hidden = self.heavy_ab_fc(heavy_ab_trans)
        ab_hidden = t.cat([light_ab_hidden, heavy_ab_hidden], axis = 1)
        return ab_hidden

    def forward_virus(self, virus, pngs_mask, ab_hidden):
        virus_and_pngs = t.cat([virus, pngs_mask], axis = 2)
        self.virus_gru.flatten_parameters()
        ab_hidden_clone = t.clone(ab_hidden)
        ab_hidden_bidirectional = t.stack((ab_hidden, ab_hidden_clone), axis = 0)
        virus_ab_all_output, _ = self.virus_gru(virus_and_pngs, ab_hidden_bidirectional)
        virus_output = virus_ab_all_output[:, -1]
        virus_output = self.fc_dropout(virus_output)
        return self.sigmoid(self.fully_connected(virus_output).squeeze())

    def forward(self, ab_light, ab_heavy, virus, pngs_mask):
        batch_size = len(ab_light)
        ab_light, ab_heavy, virus = self.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
        ab_hidden = self.forward_antibodyes(ab_light, ab_heavy, batch_size)
        return self.forward_virus(virus, pngs_mask, ab_hidden)

def get_FC_TRANS_GRU_model(conf):
    model = FC_TRANS_GRU(conf, get_embeding_matrix()).to(device)
    model = t.nn.DataParallel(model)
    return model
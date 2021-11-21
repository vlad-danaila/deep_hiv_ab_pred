from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, amino_props, amino_props_and_one_hot
from deep_hiv_ab_pred.global_constants import EMBEDDING
from deep_hiv_ab_pred.util.tools import to_torch
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM

class FC_GRU_ATT(t.nn.Module):

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
        self.light_ab_fc = t.nn.Linear(LIGHT_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.light_ab_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.light_ab_att = t.nn.Linear(LIGHT_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.light_ab_att_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.heavy_ab_fc = t.nn.Linear(HEAVY_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.heavy_ab_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
        self.heavy_ab_att = t.nn.Linear(HEAVY_ANTIBODY_TRIM * self.embeding_size, conf['RNN_HIDDEN_SIZE'])
        self.heavy_ab_att_dropout = t.nn.Dropout(conf['ANTIBODIES_DROPOUT'])
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
        return t.zeros(self.conf['NB_LAYERS'] * 2, batch_size, self.VIRUS_RNN_HIDDEN_SIZE, device=device)

    def forward_embeddings(self, ab_light, ab_heavy, virus, batch_size):
        ab_light = self.aminoacid_embedding(ab_light).reshape(batch_size, -1)
        ab_heavy = self.aminoacid_embedding(ab_heavy).reshape(batch_size, -1)
        virus = self.aminoacid_embedding(virus).reshape(batch_size, -1, self.conf['KMER_LEN_VIRUS'] * self.embeding_size)
        ab_light = self.embedding_dropout(ab_light)
        ab_heavy = self.embedding_dropout(ab_heavy)
        virus = self.embedding_dropout(virus)
        return ab_light, ab_heavy, virus

    def forward_antibodyes(self, ab_light, ab_heavy):
        light_ab_att = t.sigmoid(self.light_ab_att_dropout(self.light_ab_att(ab_light)))
        light_ab_hidden = light_ab_att * self.light_ab_dropout(self.light_ab_fc(ab_light))
        heavy_ab_att = t.sigmoid(self.heavy_ab_att_dropout(self.heavy_ab_att(ab_heavy)))
        heavy_ab_hidden = heavy_ab_att * self.heavy_ab_dropout(self.heavy_ab_fc(ab_heavy))
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
        ab_hidden = self.forward_antibodyes(ab_light, ab_heavy)
        return self.forward_virus(virus, pngs_mask, ab_hidden)

def get_FC_GRU_ATT_model(conf, embeding_type = EMBEDDING):
    if embeding_type == 'LEARNED':
        embedding_matrix = None
    elif embeding_type == 'ONE-HOT':
        embedding_matrix = t.eye(aminoacids_len)
    elif embeding_type == 'ONE-HOT-AND-PROPS':
        embedding_matrix = to_torch(amino_props_and_one_hot().values)
    elif embeding_type == 'PROPS-ONLY':
        embedding_matrix = to_torch(amino_props.values)
    else:
        raise 'The embedding type must have a valid value.'
    model = FC_GRU_ATT(conf, embedding_matrix).to(device)
    model = t.nn.DataParallel(model)
    return model
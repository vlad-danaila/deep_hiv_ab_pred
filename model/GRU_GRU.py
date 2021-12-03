from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix

class GRU_GRU(t.nn.Module):

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
        self.light_ab_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_ANTB'] * self.embeding_size,
            hidden_size = conf['RNN_HIDDEN_SIZE'],
            num_layers = 1,
            dropout = conf['ANTIBODIES_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.heavy_ab_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_ANTB'] * self.embeding_size,
            hidden_size = conf['RNN_HIDDEN_SIZE'],
            num_layers = 1,
            dropout = conf['ANTIBODIES_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.VIRUS_RNN_HIDDEN_SIZE = conf['RNN_HIDDEN_SIZE'] * 2
        self.virus_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_VIRUS'] * self.embeding_size + conf['KMER_LEN_VIRUS'],
            hidden_size = self.VIRUS_RNN_HIDDEN_SIZE,
            num_layers = 1,
            dropout = conf['VIRUS_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.embedding_dropout = t.nn.Dropout(conf['EMBEDDING_DROPOUT'])
        self.fc_dropout = t.nn.Dropout(conf['FULLY_CONNECTED_DROPOUT'])
        self.fully_connected = t.nn.Linear(2 * self.VIRUS_RNN_HIDDEN_SIZE, 1)
        self.sigmoid = t.nn.Sigmoid()

    def ab_light_state_init(self, batch_size):
        return t.zeros(2, batch_size, self.conf['RNN_HIDDEN_SIZE'], device=device)

    def ab_heavy_state_init(self, batch_size):
        return t.zeros(2, batch_size, self.conf['RNN_HIDDEN_SIZE'], device=device)

    def virus_state_init(self, batch_size):
        return t.zeros(2, batch_size, self.VIRUS_RNN_HIDDEN_SIZE, device=device)

    def forward_embeddings(self, ab_light, ab_heavy, virus, batch_size):
        ab_light = self.aminoacid_embedding(ab_light).reshape(batch_size, -1, self.conf['KMER_LEN_ANTB'] * self.embeding_size)
        ab_heavy = self.aminoacid_embedding(ab_heavy).reshape(batch_size, -1, self.conf['KMER_LEN_ANTB'] * self.embeding_size)
        virus = self.aminoacid_embedding(virus).reshape(batch_size, -1, self.conf['KMER_LEN_VIRUS'] * self.embeding_size)
        ab_light = self.embedding_dropout(ab_light)
        ab_heavy = self.embedding_dropout(ab_heavy)
        virus = self.embedding_dropout(virus)
        return ab_light, ab_heavy, virus

    def forward_antibodyes(self, ab_light, ab_heavy, batch_size):
        self.light_ab_gru.flatten_parameters()
        light_ab_all_output, light_ab_hidden = self.light_ab_gru(ab_light, self.ab_light_state_init(batch_size))
        self.heavy_ab_gru.flatten_parameters()
        heavy_ab_all_output, heavy_ab_hidden = self.heavy_ab_gru(ab_heavy, self.ab_heavy_state_init(batch_size))
        ab_hidden = t.cat([light_ab_hidden, heavy_ab_hidden], axis = 2)
        return ab_hidden

    def forward_virus(self, virus, pngs_mask, ab_hidden):
        virus_and_pngs = t.cat([virus, pngs_mask], axis = 2)
        self.virus_gru.flatten_parameters()
        virus_ab_all_output, _ = self.virus_gru(virus_and_pngs, ab_hidden)
        virus_output = virus_ab_all_output[:, -1]
        virus_output = self.fc_dropout(virus_output)
        return self.sigmoid(self.fully_connected(virus_output).squeeze())

    def forward(self, ab_light, ab_heavy, virus, pngs_mask):
        batch_size = len(ab_light)
        ab_light, ab_heavy, virus = self.forward_embeddings(ab_light, ab_heavy, virus, batch_size)
        ab_hidden = self.forward_antibodyes(ab_light, ab_heavy, batch_size)
        return self.forward_virus(virus, pngs_mask, ab_hidden)

def get_GRU_GRU_model(conf):
    model = GRU_GRU(conf, get_embeding_matrix()).to(device)
    model = t.nn.DataParallel(model)
    return model
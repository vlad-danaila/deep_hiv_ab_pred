from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.sequences import aminoacids_len

class ICERI2021Net_V2(t.nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = conf['EMBEDDING_SIZE'])
        self.light_ab_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_ANTB'] * conf['EMBEDDING_SIZE'],
            hidden_size = conf['RNN_HIDDEN_SIZE'],
            num_layers = conf['NB_LAYERS'],
            dropout = conf['ANTIBODIES_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.heavy_ab_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_ANTB'] * conf['EMBEDDING_SIZE'],
            hidden_size = conf['RNN_HIDDEN_SIZE'],
            num_layers = conf['NB_LAYERS'],
            dropout = conf['ANTIBODIES_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.VIRUS_RNN_HIDDEN_SIZE = conf['RNN_HIDDEN_SIZE'] * 2
        self.virus_gru = t.nn.GRU(
            input_size = conf['KMER_LEN_VIRUS'] * conf['EMBEDDING_SIZE'] + conf['KMER_LEN_VIRUS'],
            hidden_size = self.VIRUS_RNN_HIDDEN_SIZE,
            num_layers = conf['NB_LAYERS'],
            dropout = conf['VIRUS_RNN_DROPOUT'],
            batch_first = True,
            bidirectional = True
        )
        self.embedding_dropout = t.nn.Dropout(conf['EMBEDDING_DROPOUT'])
        self.fc_dropout = t.nn.Dropout(conf['FULLY_CONNECTED_DROPOUT'])
        self.fully_connected = t.nn.Linear(2 * self.VIRUS_RNN_HIDDEN_SIZE, 1)
        self.sigmoid = t.nn.Sigmoid()

    def ab_light_state_init(self, batch_size):
        return t.zeros(self.conf['NB_LAYERS'] * 2, batch_size, self.conf['RNN_HIDDEN_SIZE'], device=device)

    def ab_heavy_state_init(self, batch_size):
        return t.zeros(self.conf['NB_LAYERS'] * 2, batch_size, self.conf['RNN_HIDDEN_SIZE'], device=device)

    def virus_state_init(self, batch_size):
        return t.zeros(self.conf['NB_LAYERS'] * 2, batch_size, self.VIRUS_RNN_HIDDEN_SIZE, device=device)

    def forward(self, ab_light, ab_heavy, virus, pngs_mask):
        batch_size = len(ab_light)

        ab_light = self.aminoacid_embedding(ab_light).reshape(batch_size, -1, self.conf['KMER_LEN_ANTB'] * self.conf['EMBEDDING_SIZE'])
        ab_heavy = self.aminoacid_embedding(ab_heavy).reshape(batch_size, -1, self.conf['KMER_LEN_ANTB'] * self.conf['EMBEDDING_SIZE'])
        virus = self.aminoacid_embedding(virus).reshape(batch_size, -1, self.conf['KMER_LEN_VIRUS'] * self.conf['EMBEDDING_SIZE'])

        ab_light = self.embedding_dropout(ab_light)
        ab_heavy = self.embedding_dropout(ab_heavy)
        virus = self.embedding_dropout(virus)

        virus_and_pngs = t.cat([virus, pngs_mask], axis = 2)

        light_ab_all_output, light_ab_hidden = self.light_ab_gru(ab_light, self.ab_light_state_init(batch_size))
        light_ab_output = light_ab_all_output[:, -1]

        heavy_ab_all_output, heavy_ab_hidden = self.heavy_ab_gru(ab_heavy, self.ab_heavy_state_init(batch_size))
        heavy_ab_output = heavy_ab_all_output[:, -1]

        ab_hidden = t.cat([light_ab_hidden, heavy_ab_hidden], axis = 2)

        virus_ab_all_output, _ = self.virus_gru(virus_and_pngs, ab_hidden)
        virus_output = virus_ab_all_output[:, -1]

        virus_output = self.fc_dropout(virus_output)

        return self.sigmoid(self.fully_connected(virus_output).squeeze())
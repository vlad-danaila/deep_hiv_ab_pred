from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.constants import CDR_LENGHTS
from deep_hiv_ab_pred.global_constants import INCLUDE_CDR_MASK_FEATURES, INCLUDE_CDR_POSITION_FEATURES, OUTPUT_AGGREGATE_MODE
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix
from deep_hiv_ab_pred.model.PositionalEmbedding import get_positional_embeding
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index

class TRANSF(t.nn.Module):

    def __init__(self, conf, src_seq_len, tgt_seq_len, embeddings_matrix = None):
        super().__init__()
        self.conf = conf
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        if embeddings_matrix is None:
            self.embeding_size = conf['EMBEDDING_SIZE']
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
        else:
            self.embeding_size =  embeddings_matrix.shape[1]
            self.aminoacid_embedding = t.nn.Embedding(num_embeddings = aminoacids_len, embedding_dim = self.embeding_size)
            self.aminoacid_embedding.load_state_dict({'weight': embeddings_matrix})
            self.aminoacid_embedding.weight.requires_grad = False

        self.embedding_dropout = t.nn.Dropout(conf['EMBEDDING_DROPOUT'])

        transf_enc_layer = t.nn.TransformerEncoderLayer(
            # 1 is added to have the same size as the virus embed
            # the tensor is filled with zeros along this last added dimension
            d_model = self.embeding_size + conf['POS_EMBED'] + 1,
            nhead = conf['N_HEADS_ENCODER'],
            dim_feedforward = conf['TRANS_HIDDEN_ENCODER'],
            dropout = conf['TRANS_DROPOUT_ENCODER'],
            batch_first = True
        )
        self.transf_encoder = t.nn.TransformerEncoder(transf_enc_layer, conf['TRANSF_ENCODER_LAYERS'])

        trasnf_dec_layer = t.nn.TransformerDecoderLayer(
            # 1 is from the pngs mask
            d_model = self.embeding_size + conf['POS_EMBED'] + 1,
            nhead = conf['N_HEADS_DECODER'],
            dim_feedforward = conf['TRANS_HIDDEN_DECODER'],
            dropout = conf['TRANS_DROPOUT_DECODER'],
            batch_first = True
        )
        self.transf_decoder = t.nn.TransformerDecoder(trasnf_dec_layer, conf['TRANSF_DECODER_LAYERS'])

        self.fc_dropout = t.nn.Dropout(conf['FULLY_CONNECTED_DROPOUT'])
        self.fully_connected = t.nn.Linear((self.embeding_size + conf['POS_EMBED'] + 1) * tgt_seq_len, 1)
        self.sigmoid = t.nn.Sigmoid()

    def forward_embeddings(self, ab, virus, pngs_mask, batch_size):
        ab = self.embedding_dropout(self.aminoacid_embedding(ab))
        virus = self.embedding_dropout(self.aminoacid_embedding(virus))

        ab_pos_embed = get_positional_embeding(self.conf['POS_EMBED'], self.src_seq_len).repeat((batch_size, 1, 1))
        virus_pos_embed = get_positional_embeding(self.conf['POS_EMBED'], self.tgt_seq_len).repeat((batch_size, 1, 1))

        pngs_mask = pngs_mask.unsqueeze(dim = 2)
        empty = t.zeros((batch_size, self.src_seq_len, 1)).to(device)
        ab = t.cat((ab, ab_pos_embed, empty), dim = 2)
        virus = t.cat((virus, virus_pos_embed, pngs_mask), dim = 2)

        return ab, virus

    def forward_antibodyes(self, ab, ab_mask):
        return self.transf_encoder(ab, src_key_padding_mask = ab_mask)

    def forward_virus(self, virus, ab_hidden, virus_mask, ab_mask, batch_size):
        transf_out = self.transf_decoder(
            virus, ab_hidden, tgt_key_padding_mask = virus_mask, memory_key_padding_mask = ab_mask)
        transf_out = transf_out.reshape(batch_size, -1)
        transf_out = self.fc_dropout(transf_out)
        return self.sigmoid(self.fully_connected(transf_out).squeeze())

    def forward(self, ab, virus, pngs_mask):
        batch_size = ab.shape[0]
        ab_mask = ab == amino_to_index['X']
        virus_mask = virus == amino_to_index['X']
        ab, virus = self.forward_embeddings(ab, virus, pngs_mask, batch_size)
        ab_hidden = self.forward_antibodyes(ab, ab_mask)
        return self.forward_virus(virus, ab_hidden, virus_mask, ab_mask, batch_size)

def get_TRANSF_model(conf, src_seq_len, tgt_seq_len):
    model = TRANSF(conf, src_seq_len, tgt_seq_len, get_embeding_matrix()).to(device)
    model = t.nn.DataParallel(model)
    return model
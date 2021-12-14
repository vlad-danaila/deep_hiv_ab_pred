from deep_hiv_ab_pred.util.tools import device
import torch as t
from deep_hiv_ab_pred.preprocessing.constants import CDR_LENGHTS
from deep_hiv_ab_pred.global_constants import INCLUDE_CDR_MASK_FEATURES, INCLUDE_CDR_POSITION_FEATURES, OUTPUT_AGGREGATE_MODE
from deep_hiv_ab_pred.preprocessing.aminoacids import aminoacids_len, get_embeding_matrix
from deep_hiv_ab_pred.model.PositionalEmbedding import get_positional_embeding

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
            d_model = self.embeding_size + conf['POS_EMBED_LEN_ENC'],
            nhead = conf['N_HEADS_ENCODER'],
            dim_feedforward = conf['TRANS_HIDDEN_ENCODER'],
            dropout = conf['TRANS_DROPOUT_ENCODER'],
            batch_first = True
        )
        self.transf_encoder = t.nn.TransformerEncoder(transf_enc_layer, conf['TRANSF_ENCODER_LAYERS'])

        trasnf_dec_layer = t.nn.TransformerDecoderLayer(
            d_model = self.embeding_size + conf['POS_EMBED_LEN_DEC'],
            nhead = conf['N_HEADS_DECODER'],
            dim_feedforward = conf['TRANS_HIDDEN_DECODER'],
            dropout = conf['TRANS_DROPOUT_DECODER'],
            batch_first = True
        )
        self.transf_decoder = t.nn.TransformerDecoder(trasnf_dec_layer, conf['TRANSF_DECODER_LAYERS'])

        self.fc_dropout = t.nn.Dropout(conf['FULLY_CONNECTED_DROPOUT'])
        self.fully_connected = t.nn.Linear(self.embeding_size * tgt_seq_len, 1)
        self.sigmoid = t.nn.Sigmoid()

    def forward_embeddings(self, ab, virus, pngs_mask):
        ab = self.embedding_dropout(self.aminoacid_embedding(ab))
        virus = self.embedding_dropout(self.aminoacid_embedding(virus))

        ab_pos_embed = get_positional_embeding(self.conf['POS_EMBED_LEN_ENC'], self.src_seq_len)\
            .repeat((self.conf['BATCH_SIZE'], 1, 1))
        virus_pos_embed = get_positional_embeding(self.conf['POS_EMBED_LEN_DEC'], self.tgt_seq_len)\
            .repeat((self.conf['BATCH_SIZE'], 1, 1))

        pngs_mask = pngs_mask.unsqueeze(dim = 2)
        ab = t.cat((ab, ab_pos_embed), dim = 2)
        virus = t.cat((virus, pngs_mask, virus_pos_embed), dim = 2)

        return ab, virus

    def forward_antibodyes(self, ab):
        # TODO mask fac un tensor boolean care sa imi zica unde e padding
        # Vezi in documentatie ce format, ce tip de date suporta
        # Si vezi care tb sa fie true si care false, ca se poate sa fie pe dos
        return self.transf_encoder(ab)

    def forward(self, ab, virus, pngs_mask):
        ab, virus = self.forward_embeddings(ab, virus, pngs_mask)
        # TODO add masks
        ab_hidden = self.forward_antibodyes(ab)
        return self.forward_virus(virus, ab_hidden)

def get_TRANSF_model(conf, src_seq_len, tgt_seq_len):
    model = TRANSF(conf, src_seq_len, tgt_seq_len, get_embeding_matrix()).to(device)
    model = t.nn.DataParallel(model)
    return model
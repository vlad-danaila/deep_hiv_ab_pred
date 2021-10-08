import os
import collections
from util.tools import read_fasta, dump_json
from constants import RAWI_DATA, SEQ_FOLDER

if __name__ == '__main__':

    data = collections.defaultdict(lambda: [])

    for fasta_file in os.listdir(SEQ_FOLDER):
        antibody_name = fasta_file.split('_')[0]
        for seq_record in read_fasta(os.path.join(SEQ_FOLDER, fasta_file)):
            data[antibody_name].append(seq_record.id)

    dump_json(data, RAWI_DATA)
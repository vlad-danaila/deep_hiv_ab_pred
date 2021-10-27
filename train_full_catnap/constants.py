from os.path import join
from deep_hiv_ab_pred.catnap.constants import CATNAP_FOLDER

# For linux
# CLUSTALW_FOLDER = join('deep_hiv_ab_pred', 'train_full_catnap', 'clustalw')
# CLUSTALW_DOWNLOAD = 'http://www.clustal.org/download/current/clustalw-2.1-linux-x86_64-libcppstatic.tar.gz'
# CLUSTALW_FILE_ZIP = join(CLUSTALW_FOLDER, 'clustalw-2.1-linux-x86_64-libcppstatic.tar.gz')
# CLUSTALW_FILE = join(CLUSTALW_FOLDER, 'clustalw-2.1-linux-x86_64-libcppstatic')
# CLUSTALW_EXE_FILE = '/content/clustalw/clustalw-2.1-linux-x86_64-libcppstatic/clustalw2'

'''
For windows: install clustalw2, align the catnap viruses and create a phylogenetic tree
./clustalw2.exe -INFILE='c:/DOC/Workspace/HIV Article/deep_hiv_ab_pred/catnap/catnap_data/virseqs_aa.fasta' -TREE -OUTPUTTREE=dist
'''

VIRUS_TREE = join(CATNAP_FOLDER, 'virseqs_aa.ph')
VIRUS_DISTANCE_MATRIX = join(CATNAP_FOLDER, 'virseqs_aa.dst')

NB_VIRUS_CLUSTERS = 10

TRAIN_VAL_SPLIT = 1/9
TEST_SPLIT = .1

ROOT = join('deep_hiv_ab_pred', 'train_full_catnap')
SPLITS_HOLD_OUT_ONE_CLUSTER = join(ROOT, 'splits_hold_out_one_cluster.json')
MODELS_FOLDER = join(ROOT, 'models')
BEST_PARAMS = join(ROOT, 'best_parameters.json')
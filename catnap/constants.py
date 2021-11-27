import os
from os.path import join

# Folder to download the dataset
CATNAP_FOLDER = os.path.join('deep_hiv_ab_pred', 'catnap', 'catnap_data')

# Download file containing experiments data
ASSAY_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/assay.txt'
ASSAY_FILE = os.path.join(CATNAP_FOLDER, 'assay.text')

# Download file containing the aminoacid sequences of the heavy chain of the antibodies.
ANTIBODIES_HEAVY_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/heavy_seqs_aa.fasta'
ANTIBODIES_HEAVY_FASTA_FILE = os.path.join(CATNAP_FOLDER, 'heavy_seqs_aa.fasta')

# Download file containing the aminoacid sequences of the light chain of the antibodies.
ANTIBODIES_LIGHT_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/light_seqs_aa.fasta'
ANTIBODIES_LIGHT_FASTA_FILE = os.path.join(CATNAP_FOLDER, 'light_seqs_aa.fasta')

# Download file containing the aminoacid sequences of the virus envelope.
VIRUS_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/virseqs_aa.fasta'
VIRUS_FILE = os.path.join(CATNAP_FOLDER, 'virseqs_aa.fasta')

# PNGS stands for 'potential N-linked glycosylation site'. It is a specific region of the virus envelope where a carbohydrate is attached (https://en.wikipedia.org/wiki/N-linked_glycosylation).
# The PNGS presence or absence in specific regions of the virus envelope is crucial because it can alter the antibody binding capabilities.
VIRUS_WITH_PNGS_DONLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/virseqs_aa_O.fasta'
VIRUS_WITH_PNGS_FILE = os.path.join(CATNAP_FOLDER, 'virseqs_aa_O.fasta')

# Download file containing antibodies details
ANTIBODIES_DETAILS_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/abs.txt'
ANTIBODIES_DETAILS_FILE = os.path.join(CATNAP_FOLDER, 'antibodies_details.text')

CATNAP_FLAT = os.path.join('deep_hiv_ab_pred', 'catnap', 'catnap_flat.json')
IC50_TRESHOLD = 50

# CDR
CDR_FOLDER = os.path.join('deep_hiv_ab_pred', 'catnap', 'cdr_data')
PARATOME_AB_LIGHT_CDR = join(CDR_FOLDER, 'Paratome output ab light.txt')
PARATOME_AB_HEAVY_CDR = join(CDR_FOLDER, 'Paratome output ab heavy.txt')
ABRSA_AB_LIGHT_CDR = join(CDR_FOLDER, 'AbRSA output ab light.txt')
ABRSA_AB_HEAVY_CDR = join(CDR_FOLDER, 'AbRSA output ab heavy.txt')
AB_CDRS = join(CDR_FOLDER, 'CDRs.json')

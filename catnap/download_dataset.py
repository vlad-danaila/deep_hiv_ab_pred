from util.tools import create_folder, download_file
import os

# Folder to download the dataset
CATNAP_FOLDER = 'catnap_data'

# Download file containing experiments data
ASSAY_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/assay.txt'
ASSAY_FILE = os.path.join(CATNAP_FOLDER, 'assay.text')

# Download file containing the aminoacid sequences of the heavy chain of the antibodies.
ANTIBODIES_HEAVY_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/heavy_seqs_aa.fasta'
ANTIBODIES_HEAVY_FILE = os.path.join(CATNAP_FOLDER, 'heavy_seqs_aa.fasta' )

# Download file containing the aminoacid sequences of the light chain of the antibodies.
ANTIBODIES_LIGHT_DOWNLOAD = 'https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?/scratch/NEUTRALIZATION/light_seqs_aa.fasta'
ANTIBODIES_LIGHT_FILE = os.path.join(CATNAP_FOLDER, 'light_seqs_aa.fasta')

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

create_folder(CATNAP_FOLDER)
download_file(ASSAY_DOWNLOAD, ASSAY_FILE)
download_file(ANTIBODIES_HEAVY_DOWNLOAD, ANTIBODIES_HEAVY_FILE)
download_file(ANTIBODIES_LIGHT_DOWNLOAD, ANTIBODIES_LIGHT_FILE)
download_file(VIRUS_DOWNLOAD, VIRUS_FILE)
download_file(VIRUS_WITH_PNGS_DONLOAD, VIRUS_WITH_PNGS_FILE)
download_file(ANTIBODIES_DETAILS_DOWNLOAD, ANTIBODIES_DETAILS_FILE)

""" The virus sequences are aligned, sometimes the characters # and * are inserted by alignment software. 
Unfortunately, we are unsure about the exact meaning of these characters, and they could be software specific. 
For this reason, we are removing them. """
def removeHashes(path):
    with open(path, 'r') as file:
        content = file.read()
        content = content.replace('#', '')
        content = content.replace('*', '')
    with open(path, 'w') as file:
        file.write(content)

removeHashes(VIRUS_FILE)



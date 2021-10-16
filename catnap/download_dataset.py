from util.tools import create_folder, download_file
from catnap.constants import *

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

def download_catnap():
    create_folder(CATNAP_FOLDER)
    download_file(ASSAY_DOWNLOAD, ASSAY_FILE)
    download_file(ANTIBODIES_HEAVY_DOWNLOAD, ANTIBODIES_HEAVY_FILE)
    download_file(ANTIBODIES_LIGHT_DOWNLOAD, ANTIBODIES_LIGHT_FILE)
    download_file(VIRUS_DOWNLOAD, VIRUS_FILE)
    download_file(VIRUS_WITH_PNGS_DONLOAD, VIRUS_WITH_PNGS_FILE)
    download_file(ANTIBODIES_DETAILS_DOWNLOAD, ANTIBODIES_DETAILS_FILE)
    removeHashes(VIRUS_FILE)

if __name__ == '__main__':
    download_catnap()


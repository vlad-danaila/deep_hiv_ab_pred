from deep_hiv_ab_pred.util.tools import download_file, create_folder
from deep_hiv_ab_pred.train_full_catnap.constants import *
from Bio import Phylo

def print_phylogenetic_tree():
    tree = Phylo.read(VIRUS_TREE, "newick")
    tree.ladderize()
    Phylo.draw_ascii(tree)

if __name__ == '__main__':
    # For making sure the phylogenetic tree was created
    # print_phylogenetic_tree()
    pass
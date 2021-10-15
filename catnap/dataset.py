import collections
from catnap.constants import CATNAP_FLAT, IC50_TRESHOLD
import pandas as pd
from catnap.download_dataset import ASSAY_FILE
from catnap.censored_data_preprocess import estimate_censored_mean
from tqdm import tqdm
from util.tools import dump_json, read_yaml
from preprocessing.sequences import parse_catnap_sequences
from hyperparameters.constants import CONF_ICERI

KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'

def read_assays_df():
    assay_df = pd.read_csv(ASSAY_FILE, sep = '\t')
    # Drop unnecssary columns
    assay_df = assay_df.drop(['Reference', 'Pubmed ID', 'IC80', 'ID50'], axis = 1)
    # Exclude mixed(contains a '+' character) and polyclonal antibodeis assays
    assay_df = assay_df[~assay_df['Antibody'].str.contains("[Pp]olyclonal|\+")]
    # Filter null and nan
    assay_df = assay_df[assay_df['IC50'].notnull()]
    return assay_df

def virus_is_sensitive(ic50: str):
    if ic50.startswith('>'):
        return False
    ic50 = float(ic50[1:]) if ic50.startswith('<') else float(ic50)
    return ic50 < IC50_TRESHOLD

def catnap_by_antibodies():
    assays_df = read_assays_df()
    assays = collections.defaultdict(lambda: {})
    conf = read_yaml(CONF_ICERI)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
    )
    # The for loop iterates through assays grouped by the antibody and virus pairs
    for (antibody, virus), df in tqdm(assays_df.groupby(['Antibody', 'Virus'])):
        if antibody not in antibody_light_seq or antibody not in antibody_heavy_seq or virus not in virus_seq:
            continue
        ic50 = df.IC50
        # consider resistant by default
        outcome = False
        is_sensitive = ic50.apply(virus_is_sensitive).values
        # if all measurements result in sensitive
        if is_sensitive.sum() == len(is_sensitive):
            outcome = True
        # if some measurements result in sensitive and some in resistant
        # estimate the mean taking into account the censored data
        elif is_sensitive.sum() > 0:
            single_valued = ic50.apply(lambda x: not (x.startswith('>') or x.startswith('<')))
            right_censored = ic50.apply(lambda x: x.startswith('>'))
            left_censored = ic50.apply(lambda x: x.startswith('<'))
            estimated_censored_mean = estimate_censored_mean(
                single_valued = [float(x) for x in ic50[single_valued]],
                left_censored = [float(x[1:]) for x in ic50[left_censored]],
                right_censored = [float(x[1:]) for x in ic50[right_censored]]
            )
            outcome = estimated_censored_mean < 50
        assays[antibody][virus] = outcome
    return assays

def flatten_catnap_data(catnap_data):
    flat = []
    id = 0
    for antibody, viruses in catnap_data.items():
        for virus in viruses:
            flat.append((id, antibody, virus, catnap_data[antibody][virus]))
            id += 1
    return flat

if __name__ == '__main__':
    catnap_by_antibodies = catnap_by_antibodies()
    catnap_flat = flatten_catnap_data(catnap_by_antibodies)
    dump_json(catnap_flat, CATNAP_FLAT)


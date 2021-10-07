import collections

import pandas as pd
from catnap.download_dataset import ASSAY_FILE
from catnap.censored_data_preprocess import estimate_censored_mean
from tqdm import tqdm
from util.tools import dump_json

RAWI_DATA = 'compare_to_Rawi_gbm/Rawi_data.json'
CATNAP_ASSAYS_CLASSIFICATION = 'catnap_classification.json'
IC50_TRESHOLD = 50

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

def read_assays_for_classification():
    assays_df = read_assays_df()
    assays = collections.defaultdict(lambda: {})
    # The for loop iterates through assays grouped by the antibody and virus pairs
    for (antibody, virus), df in tqdm(assays_df.groupby(['Antibody', 'Virus'])):
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

if __name__ == '__main__':
    assays = read_assays_for_classification()
    dump_json(assays, CATNAP_ASSAYS_CLASSIFICATION)


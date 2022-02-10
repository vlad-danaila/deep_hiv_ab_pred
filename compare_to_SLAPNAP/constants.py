from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import ANTIBODIES_LIST as RAWI_ABS

SLAPNAP_ABS = [ "2G12", "PG16", "PG9", "PGT145", "PGDM1400", "VRC26.08", "VRC26.25", "PGT128", "10-1074", "10-996", "PGT121", "VRC38.01", "PGT135",
                    "DH270.1", "DH270.5", "DH270.6", "VRC01", "3BNC117", "VRC-PG04", "NIH45-46", "VRC03", "VRC-CH31", "CH01", "HJ16", "VRC07",
                    "b12", "PGT151", "VRC34.01", "8ANC195", "35O22", "2F5", "4E10" ]

# '10-996' is only present in SLAPNAP but not in Rawi GBM
# 'VRC13' is only present in Rawi GBM but not in SLAPNAP
# {'VRC29.03', 'VRC13'} were not taken into account because they did not have the sequences available in CATNAP
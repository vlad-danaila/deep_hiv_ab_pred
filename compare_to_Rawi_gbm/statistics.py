from util.tools import read_json_file
import numpy as np

RAWI_DATA = 'Rawi_data.json'

if __name__ == '__main__':
    rawi_data = read_json_file(RAWI_DATA)
    for antibody, viruses in rawi_data.items():
        print(antibody, len(viruses))



    # rawi_data_size = np.array([len(viruses) for viruses in rawi_data.values()])

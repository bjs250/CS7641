import pandas as pd

def preprocess():
    DATA_FILENAME = "data/cmc-data.csv"
    data_headers = ["Wife's Age", "Wife's Education", "Husband's Education", "Number of children ever born", "Wife's religion", "Wife's now working?", "Husbands's occupation", "Standard-of-living index", "Media exposure", "Contraceptive method used"]
    df = pd.read_csv(DATA_FILENAME, sep=',', names=data_headers)
    return df
import pandas as pd
from matplotlib import pyplot as plt

import matplotlib

matplotlib.use("agg")

from pathlib import Path


# Expects a pandas dataframe with the following columns:
#  - random (bool): the record was in the random sample run-in phase
#  - label (bool): the record was a include/exclude
#  - order (int): how many records were seen before this one
def assess_non_nice(df):
    if 'random' in df.columns:
        df = df[df['random'] == False]
    time_to_find = df[df['label'] == True]['order']
    # calculate lags between finds
    time_lags = time_to_find.diff()
    # the rolling average of the previous finds
    avg_preceding = time_lags.rolling(5, center=False).mean().shift()
    # count the proportion where the time lag drops significantly
    return sum(avg_preceding > time_lags * 4) / avg_preceding.count()


# Expects a pandas dataframe with the following columns:
#  - random (bool): the record was in the random sample run-in phase
#  - label (bool): the record was a include/exclude
def assess_gain(df):
    if 'random' in df.columns:
        df = df[df['random'] == False]
    # calculate the expected slope if screening randomly
    total_records = df['label'].count()
    total_found = df['label'].sum()
    slope = total_found / total_records
    # calculate the actual and expected found at each record screened
    n_found = df['label'].cumsum()
    exp_found = [i * slope for i in range(1, total_records + 1)]
    # sum proportion found at each record screened and normalize by number of records
    return sum((n_found - exp_found) / total_found) / total_records


for file in Path('./data/rankings').glob('*.feather'):
    df = pd.read_feather(file)
    df = df[df['random'] == False]
    df = df.filter(['order', 'label'])
    df['cumulative'] = df['label'].cumsum()
    plt.clf()
    plt.plot(df['order'], df['cumulative'])
    plt.suptitle(
        "Gain: {gain:.3f}; stepping: {stepping:.3f}".format(gain=assess_gain(df), stepping=assess_non_nice(df)))
    plt.savefig('./data/rankings/plots/' + file.name.replace(".feather", ".png"))

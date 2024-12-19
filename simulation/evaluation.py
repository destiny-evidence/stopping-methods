import pandas as pd

from shared.config import settings
import logging

logger = logging.getLogger('evaluate')

def plot_curve():
    pass
    
# my_seen_data = self.dataset.get_seen_data()  # a df showing the 'screened' data at each simulation step
# TP = my_seen_data['labels'].sum()  # the number of included records within seen data
# FP=my_seen_data.shape[0]-TP#the number of all negative records that came up during screening so far
# FN = self.n_incl - TP  # unseen positives
# TN = self.dataset.df.shape[0]-my_seen_data.shape[0]-FN # all the negatives in the unscreened data
# assert FN+TN+TP+FP == self.dataset.df.shape[0]

def evaluate(data, recall_target, method):
    stop_column ="method-{}-safe_to_stop".format(method)

    # Identify first batch to stop
    stop_index = data.index[data[stop_column] == True].tolist()
    if len(stop_index) == 0:
        return

    stop_index = stop_index[0]

    # Compute recall
    recall = data.loc[stop_index,'n_incl_seen'] / data.loc[stop_index,'n_incl']

    # cost (num_shown / num_docs)
    cost = data.loc[stop_index,'n_seen'] / data.loc[stop_index,'n_total']
    logger.info(f'{method} | proportion seen -> {cost:.2%}')

    logger.info(f'{method} | recall -> {recall}')

    # loss (amount by which achieved recall is below target recall)
    if recall >= recall_target:
        logger.info(f'{method} | loss -> 0')
    else:
        logger.info(f'{method} | loss -> {recall_target - recall}')

    #logger.debug(data.loc[stop_index])


if __name__ == '__main__':
    logger.info('Run evaluation')
    df = pd.read_csv(settings.result_data_path / 'results.csv')

    for meth in df['method'].unique():
        for dataset in df['dataset'].unique():
            logger.info(f'------- {dataset} -----------')
            evaluate(df[df['dataset']==dataset], recall_target=0.95, method=meth)

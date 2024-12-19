def plot_curve():
    pass
    
# my_seen_data = self.dataset.get_seen_data()  # a df showing the 'screened' data at each simulation step
# TP = my_seen_data['labels'].sum()  # the number of included records within seen data
# FP=my_seen_data.shape[0]-TP#the number of all negative records that came up during screening so far
# FN = self.n_incl - TP  # unseen positives
# TN = self.dataset.df.shape[0]-my_seen_data.shape[0]-FN # all the negatives in the unscreened data
# assert FN+TN+TP+FP == self.dataset.df.shape[0]

def evaluate(recall_target, method):
    stop_column ="method-{}-safe_to_stop".format(method)
    stop_index = df.index[df[stop_column] == True].tolist()[0]#check if 'TRUE' or True (ie bool or string)

    # Compute recall

    recall = float(df.loc[index,'n_incl_batch']) / float(df.loc[index,'n_incl'])

    # cost (num_shown / num_docs)
    cost = float(df.loc[index,'n_seen']) / float(df.loc[index,'n_total'])

    # loss (amount by which achieved recall is below target recall)
    if recall >= recall_target:
        loss = 0
    else:
        loss = (recall_target - recall)

Some methods assume full set of scores before and after batch
best-model strategy doesn't work here, because changing models or hyperparameters influence the score distribution in unexpected ways

Solution: 
* Remember all scores for all models (at different fixed defaults) for all datasets for all cycles for all repeats (with different init random sample)
* Needs fixed batch size (`BatchStrategy.STATIC`)
* Needs `predict_on_all = True`
* Needs `use_fine_tuning = False`
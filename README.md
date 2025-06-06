# Stopping methods for priority screening
![logo](logo.png)


Priority screening has gained popularity across all major systematic review tools.
This process uses machine learning to rank all remaining unseen documents by their relevance.
To fully benefit from the prioritisation, users need to know when it is safe to stop screening.
We define stopping methods 'safe to stop' as:
* a way to determine when a pre-defined recall is met
* a way to quantify certainty

For information how to execute the framework, check [`simulation/README.md`](simulation/README.md)

## Vision
With this project, we aim to consolidate all available stopping methods and evaluation datasets into a single repository.
We also define an standardised format to execute stopping methods in order to run rigorous evaluation.
This will allow us to compare characteristics of each method to inform guidelines on how to safely use them.

We envision this project to define an interchangeable format to allow users of digital evidence synthesis tools to use all available stopping methods in their particular review.
Further, developers of stopping criteria should adapt this format.

In the future, we hope to share evaluations for all datasets and all stopping methods on a website.

## Datasets
|    | Reference | Research area | Number of records | Number includes | Data URL                                                                                                     | Publicly available | 
|---:|:---------:|--------------:|------------------:|----------------:|--------------------------------------------------------------------------------------------------------------|--------------------|

## Evaluation framework
Degrees of freedom to validate
* ML Model characteristics
* Size of initial random sample
* Criterion-specific recall/confidence targets
* Training regime
* Ranking model hyperparameter tuning

```
for each repitition
  for each ML model type
    for each initial random sample size
      for each dataset
        for each batch in dataset ¹ ²
          train model of previsously seen data
          rank unseen data
          for each stopping criterion
            for each stopping criterion sub-config (e.g. recall target)
              computer stopping value
        compute performance metrics (precision, recall, includes missed, work saved, ...)

¹ until all criteria stop?
² adaptive batch size, e.g. growing with the number of seen records or number of includes per batch
```

## Project structure
* Datasets: Scripts to retrieve datasets for evaluation
* Stopping methods: Implementations or adapters to run a stopping criterion
* Evaluation framework: Scripts to run simulations

### How to run 
```
Requires python 3.12–3.14
```

```
# To pre-compute rankings run


# To run simulation
PYTHONPATH=. python simulation/simulate.py FIXED

# To run evaluation
PYTHONPATH=. python simulation/evaluation.py 
```

### HPC setup
```bash
cd /p/tmp/user/
git clone git@github.com:destiny-evidence/stopping-methods.git
cd stopping-methods
module load anaconda/2024.10
# verify we are really using the correct python
which python
python --version
# set up virtualenv
python -m venv data/venv
source data/venv/bin/activate
pip install -r requirements.txt

# pre-compute rankings
 PYTHONPATH=. python simulation/rank.py SLURM --models trans-rank --models svm --models lightgbm --models sgd  --models logreg \
                                --dyn-min-batch-size 25 --dyn-max-batch-size 200 --dyn-min-batch-incl 2 \
                                --num-random-init 500 --min-dataset-size 1000 --num-repeats 3 \
                                --min-inclusion-rate 0.01 --tuning-interval 4 --store-feather --slurm-user "???@pik-potsdam.de" --slurm-hours 23 --slurm-gpu
# or
 PYTHONPATH=. python simulation/rank.py SLURM --models trans-rank --models svm --models lightgbm --models sgd  --models logreg \
                                --dyn-min-batch-size 25 --dyn-max-batch-size 200 --dyn-min-batch-incl 2 \
                                --num-random-init 500 --min-dataset-size 1000 --num-repeats 3 \
                                --min-inclusion-rate 0.01 --tuning-interval 4 --store-feather --slurm-user "???@pik-potsdam.de" --slurm-hours 23
```

## Roadmap
- [ ] Collect all resources
- [ ] Collect datasets and transform them into a common format
- [ ] Collect available implementations and transform them into a common format
- [ ] Set up evaluation pipelines
- [ ] Run simulations for all datasets and methods
- [ ] Generate evaluation reports

## Additional resources
* "Cohen dataset": https://dmice.ohsu.edu/cohenaa/systematic-drug-class-review-data.html
* Dataset beyond the Cohen et al data linked to OpenAlex and enriched: https://github.com/asreview/synergy-dataset
* SYNERGY Dataset with added inclusion criteria: https://github.com/VeenDuco/inclusion_exclusion_priors
* Collection of many existing dataset collections https://github.com/WojciechKusa/systematic-review-datasets
* Rapid screening project: https://github.com/mcallaghan/rapid-screening/
* Simplified buscar score web app: https://github.com/mcallaghan/buscar-app
* LLMs for prioritisation: https://github.com/mcallaghan/ml-screening-evaluation
* buscar in R and python: https://github.com/mcallaghan/buscarR / https://github.com/mcallaghan/buscarpy

## Contributors
This repository was started at the [Evidence Synthesis Hackathon 2024](https://www.eshackathon.org/) by
* Lena Schmidt (NIHR)
* Francesca Tinsdeall (University of Edinburgh)
* Sergio Graziosi (EPPI@UCL)
* James Thomas (EPPI@UCL)
* Diana Danilenko (PIK)
* Tim Repke (PIK)


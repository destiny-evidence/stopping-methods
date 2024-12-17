# Stopping methods for priority screening
Priority screening has gained popularity across all major systematic review tools.
This process uses machine learning to rank all remaining unseen documents by their relevance.
To fully benefit from the prioritisation, users need to know when it is safe to stop screening.
We define stopping methods 'safe to stop' as:
* a way to determine when a pre-defined recall is met
* a way to quantify certainty

## Vision
With this project, we aim to consolidate all available stopping methods and evaluation datasets into a single repository.
We also define an standardised format to execute stopping methods in order to run rigorous evaluation.
This will allow us to compare characteristics of each method to inform guidelines on how to safely use them.

We envision this project to define an interchangeable format to allow users of digital evidence synthesis tools to use all available stopping methods in their particular review.
Further, developers of stopping criteria should adapt this format.

In the future, we hope to share evaluations for all datasets and all stopping methods on a website.

## Datasets
In this 

## Evaluation framework
The procedure of an evaluation run for a stopping method is as follows:
* For each iteration of the cross-validation outer loop (3 splits)
  * Train initial classifier from random set of documents
  * Hyperparameter optimisation ?
  * Rank remaining records
  * Compute stopping criterion

## Project structure
* Datasets: Scripts to retrieve datasets for evaluation
* Stopping methods: Implementations or adapters to run a stopping criterion
* Evaluation framework: Scripts to run simulations

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
* Francesca Tinsdeall (Uni Edinburgh)
* Sergio Graziosi (EPPI@UCL)
* James Thomas (EPPI@UCL)
* Diana Danilenko (PIK)
* Tim Repke (PIK)


# Hackathon and beyond todo list

## Dataset tasks
* Retrieve and share EPPI dataset
* Double-check license and sharing issues for eppi dataset
* Script to transform eppi dataset into standard CSV
* Pull in synergy dataset
* Script to transform synergy to standard csv
* Pull in Wojciech datasets
* Scripts to transform Wojciech datasets into standard csv
* Dataset descriptors (overview table of sources, basic info, licensing, etc)
* Dataset details (e.g. structured format for inclusion criteria, etc)
* A way to generate random datasets based on specific parameters, eg https://github.com/mcallaghan/buscarpy/blob/main/buscarpy/__init__.py#L26

## Documentation
* Collect all resources on the topic in this repository
* 

## Stopping simulation tasks
* Implement generic stopping interface*
* Implement stopping criteria

## Ranking simulation tasks
* Implement framework
* Implement ranking algorithms

## Evaluation
* Summarise all simulation runs per method, ranker, paramset, ...
* Produce plots from a run per dataset with vertical lines where each stopping criterion would ave first stopped

## Improvements
* Run all rankers N times for each dataset and store rankings
* Use rankings to compute stopping methods
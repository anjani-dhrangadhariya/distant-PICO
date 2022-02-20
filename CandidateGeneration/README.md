# Description

Code to generate weak PICO labels for EBM-PICO corpus. `label_generator.py` takes input EBM-PICO training set, weakly labels it using multiple labeling functions (LF) and saves the labeled candidates to local. Use `args` argument parser to set the program flow.

`level1` labels EBM-PICO training set using UMLS ontologies for selected and PICOS-mapped semantic types.
`level2` labels EBM-PICO training set with PICOS using non-UMLS ontologies all of which were downloaded from [NCBO BioPortal](https://bioportal.bioontology.org/).
`level3` labels EBM-PICO training set using Distant Supervision from clinicaltrials.org database.
`level4` labels EBM-PICO training set using several ReGeX's, heuristics and some hand-crafted dictionaries.
`level5` uses external models to label EBM-PICO training set (TODO).

Ignore the `SourceFetcher`, `TargetFetcher`, `SourceTargetExpander`, `SourceTargetAligner` directories.
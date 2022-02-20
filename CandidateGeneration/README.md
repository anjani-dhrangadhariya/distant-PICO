# Description

Code to generate weak PICO labels for EBM-PICO corpus. `label_generator.py` takes input EBM-PICO training set, weakly labels it using multiple labeling functions (LF) and saves the labeled candidates to local. Use `args` argument parser to set the program flow.

`level1` labels EBM-PICO training set using concepts from UMLS ontologies for selected, PICOS-mapped semantic types.
`level2` labels EBM-PICO training set with PICOS using non-UMLS ontologies all of which were downloaded from [NCBO BioPortal](https://bioportal.bioontology.org/). The non-UMLS ontologies used: Disease Ontology (DO), Human Phenotype Ontology (HP), Comparative Toxicogenomics Database (CTD), Ontology of Adverse Events (OAE), and Chemical Entities of Biological Interest (ChEBI).
`level3` labels EBM-PICO training set using Distant Supervision from clinicaltrials.org database.
`level4` labels EBM-PICO training set using several ReGeX's, heuristics and some hand-crafted dictionaries.
`level5` uses external models to label EBM-PICO training set (TODO).

The labeling functions label 1 wherever a term or pattern is found. UMLS LFs label -1 or abstain as well. There are no dedicated LF's to actually emit 0 label. For the tokens where an LF did not label 1 or -1, the label is set to 0 (out-of-the-span label). This still leads to high true negative rate (TNR) for almost all the LF's.

Ignore the `SourceFetcher`, `TargetFetcher`, `SourceTargetExpander`, `SourceTargetAligner` directories.
# Description

 PICOS stands for Participant, Intervention, Comparator, Outcomes, Study type. Participants (P) are the patients or healthy volunteers involved in a randomized controlled trial. Interventions include drugs, medical devices, procedures, vaccines, and other products that are being investigated in a randomized controlled trial. Interventions can also include noninvasive approaches, such as education or modifying diet and exercise. A comparator (C) is a treatment used to compare against the main intervention (I) and measure its effect. Outcomes are the measurable effects of interventions on the participants involved in the trial. PICOS is a mixed entity/span recognition task whereby a machine is required to identify text spans with information about participant characteristics like age, gender, ethnicity, disease condition, sample size in the trial. Intervention (and comparator) entity/span includes the intervention being administered, the mode of administration, and sometimes even the dosage and frequency of the administration. Outcome span/entity includes the outcomes (primary and secondary) being measured and the method or instrument of measurement.

This directory has the code to generate weak PICO labels for EBM-PICO corpus. `label_generator.py` takes input EBM-PICO training set, weakly labels it using multiple labeling functions (LF) and saves the labeled candidates to local. Use `args` argument parser to set the program flow.

`level1` labels EBM-PICO training set using concepts from UMLS ontologies for selected, PICOS-mapped semantic types.
`level2` labels EBM-PICO training set with PICOS using non-UMLS ontologies all of which were downloaded from [NCBO BioPortal](https://bioportal.bioontology.org/). The non-UMLS ontologies used: Disease Ontology (DO), Human Phenotype Ontology (HP), Comparative Toxicogenomics Database (CTD), Ontology of Adverse Events (OAE), and Chemical Entities of Biological Interest (ChEBI).
`level3` labels EBM-PICO training set using Distant Supervision from clinicaltrials.org database.
`level4` labels EBM-PICO training set using several ReGeX's, heuristics and some hand-crafted dictionaries.
`level5` uses external models to label EBM-PICO training set (TODO).

The labeling functions label 1 wherever a term or pattern is found. UMLS LFs label -1 or abstain as well. There are no dedicated LF's to actually emit 0 label. For the tokens where an LF did not label 1 or -1, the label is set to 0 (out-of-the-span label). This still leads to high true negative rate (TNR) for almost all the LF's.

Ignore the `SourceFetcher`, `TargetFetcher`, `SourceTargetExpander`, `SourceTargetAligner` directories.
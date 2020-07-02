## Overview

* iDeepC.py is the main program.
* CRIP_data.py and GraphProt_CLIP_data.py are the data processing of the specified dataset.
* CRIP_C22ORF28.h5 and GraphProt_CLIP_C22ORF28_Baltz2012.h5 are the initial model parameters when selecting the specified dataset.

## Dependency

* python3.6

* tensorflow-gpu 1.13.1

* keras 2.3.0

* skearn 0.0

* weblogo 3.7.5


## Dataset

* CRIP: RBP-37

* GraphProt_CLIP_sequence: RBP-24


## Usage

* python.py [-h] [--dataset DATASET] [--protein_name PROTEIN_NAME]

* Enter the data set and protein name. The default is the CRIP dataset and ALKBH5 protein.



## Identify motifs

* Set draw_motifs to True in the test function，You can get the .eps file of visual motifs.

* The default saved file address is source directory('./'+protein_name+'.eps').



## Contact
* xxxxx

## Reference
* xxxxx



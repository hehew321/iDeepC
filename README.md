## Overview
We present a RBP-specific method iDeepC for predicting RBP binding sites on circRNAs from sequences. iDeepC adopts a Siamese-like neural network consisting of a network module with a lightweight attention and a metric module. The network module with pre-training learns embeddings for a pair of sequences, whose embedding difference is fed to the metric module to estimate the binding potential. iDeepC is able to capture mutual information between circRNAs, and thus mitigate data scarcity for poorly characterized RBPs

## Code details
* iDeepC.py is the main program.
* RBP37_data.py.py and RBP24_data.py are the data processing script of the RBP-37 (CRIP) and RBP-24 (GraphProt) dataset, respectively.
* RBP37_C22ORF28.h5 and RBP24_C22ORF28_Baltz2012.h5 are the pretrained model parameters for C22ORF28, and they are respectively loaded when training iDeepC models for individual RBPs in RBP-37 and RBP-24 dataset.

## Dependency

* python3.6

* tensorflow-gpu 1.13.1

* keras 2.3.0

* skearn 0.20.0

* weblogo 3.7.5


## Dataset

* RBP-37: RBP-37 for RBP binding circRNAs updaed from CRIP, it consists of 37 RBP datasets, each  corresponds to one RBP. All the datasets are  at datasets/CRIP_split/ in this repository.

* RBP-24: RBP-24 for RBP binding linear RNAs, it consists of 24 datasets, each corresponds to one RBP. The whole dataset can be downloaded from  http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 


## Usage

# You can train and then test：
* python train.py [-h] [--dataset DATASET] [--protein_name PROTEIN_NAME]
* python test.py [-h] [--dataset DATASET] [--protein_name PROTEIN_NAME]

# Of course, you can also run iDeepC directly：
* python iDeepC.py [-h] [--dataset DATASET] [--protein_name PROTEIN_NAME]

  -- Specify the dataset for --dataset and protein name for --protein_name. The default value is the RBP37 dataset and ALKBH5 protein, which will train and test iDeepC model for ALKBH5 in RBP-37 dataset. 

## Identify motifs by iDeepC

* Set draw_motifs to True in the test function in iDeepC.py，you can get the .eps file of detected motifs by iDeepC.

  -- The default saved directory is source directory('./'+protein_name+'.eps').



## Contact
* 2008xypanatsjtu.edu.cn

## Reference
* xxxxx



# Deep Structural Causal Modelling of the Clinical and Radiological Phenotype of Alzheimer's Disease

This repository contains code to replicate the experiment for the NeurIPS workshop submission titled: 'Deep Structural Causal Modelling of the Clinical and Radiological Phenotype of Alzheimer's Disease'. 

The work builds on the paper 'Deep Structural Causal Models for Tractable Counterfactual Inference' by Pawlowski and Castro et al.: 

```
@inproceedings{pawlowski2020dscm,
    author = {Pawlowski, Nick and Castro, Daniel C. and Glocker, Ben},
    title = {Deep Structural Causal Models for Tractable Counterfactual Inference},
    year = {2020},
    booktitle={Advances in Neural Information Processing Systems},
}
```

and on work by Reinhold et al.: 

```
@article{Reinhold2021ASclerosis,
    title = {{A Structural Causal Model for MR Images of Multiple Sclerosis}},
    year = {2021},
    journal = {Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)},
    author = {Reinhold, Jacob C. and Carass, Aaron and Prince, Jerry L.},
    pages = {782--792},
    volume = {12905 LNCS},
    publisher = {Springer Science and Business Media Deutschland GmbH},
    url = {https://link.springer.com/chapter/10.1007/978-3-030-87240-3_75},
    isbn = {9783030872397},
    doi = {10.1007/978-3-030-87240-3{\_}75/FIGURES/6},
    issn = {16113349},
    arxivId = {2103.03158},
    keywords = {Causal inference, MRI, Multiple sclerosis}
}
```

For the original DSCM paper, please refer to the [tagged code](https://github.com/biomedia-mira/deepscm/tree/neurips_2020) for the code used for the NeurIPS publication.



## Structure
This repository contains code and assets structured as follows:

- `deepscm/`: contains the code used for running the experiments
    - `arch/`: model architectures used in experiments
    - `datasets/`: script for dataset generation and data loading used in experiments
    - `distributions/`: implementations of useful distributions or transformations
    - `experiments/`: implementation of experiments
- `SVIExperiment/`
    - `ConditionalVISEM/`: checkpoints of the trained model

## Requirements
Python 3.7.2 is used for all experiments and you will need to install the following packages:
```
pip install numpy pandas pyro-ppl pytorch-lightning scikit-image scikit-learn scipy seaborn tensorboard torch torchvision
```
or simply run `pip install -r requirements.txt`.


## Usage

We assume that the code is executed from the root directory of this repository.

### Training and evaluation 

The model can be trained using:
```
python -m deepscm.experiments.medical.trainer -e SVIExperiment -m ConditionalVISEM --default_root_dir /path/to/root/directory/ --downsample 3 --decoder_type fixed_var --train_batch_size 256 --gpus 0
```
where `/path/to/root/directory/` refers to the root directory. Note that `ConditionalVISEM` is the full model (DSCM). 

In addition, the provided checkpoints can be used for testing and plotting:
```
python -m deepscm.experiments.medical.tester -c /path/to/checkpoint/version_?
```
where `/path/to/checkpoint/version_?` refers to the path containing the specific [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) run. The notebooks for plotting are situated in [`deepscm/experiments/plotting/`](deepscm/experiments/plotting/).

### ADNI

We are unable to share the ADNI dataset. However, the research data repository is accesible following a data application here: [ADNI-data](https://adni.loni.usc.edu/data-samples/access-data/#access_data). 

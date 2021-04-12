# Advanced Topics in Machine Learning Project

## Authors

See group report

## Sources

Reproduction project of the paper: β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017

## Code Structure
### Datasets
Methods and classes that deal with dataset management (train test split), as well as noise generation methods.

### Experiments
Here files concerning experiments are defined.

  1. Files that train the models and saves the models
  2. Files with the trained models and their loss during training (under trained_models)
  3. Files that computes the entanglement metrics on the trained models
  4. Files that generate plots.

### Models
Here models are defined: Beta-VAE, Factor-VAE and Control-VAE.

### Train
Methods concerning training, such as train and test are defined. Additionally, the entanglement metrics are defined here.

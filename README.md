# Latent Linear Adjustment autoencoders: A novel method for estimating and emulating dynamic precipitation at high resolution

This repository contains the code of the autoencoder model proposed in "Latent Linear Adjustment autoencoders: A novel method for estimating and emulating dynamic precipitation at high resolution" (TODO: add arxiv/gmd link). In the manuscript, we have demonstrated how Latent Linear Adjustment autoencoders can be applied for dynamical adjustment at high resolution and for emulating dynamically-induced variability in daily precipitation fields. Futher applications such as regional detection & attribution, statistical downscaling or transfer learning between models are conceivable.

This README is not intended to be completely self-explanatory, and should be read alongside the manuscript. Below we give an overview of the model, followed by detailed instructions how to reproduce the results reported in the manuscript. 

## Model
Building on variational autoencoders, we introduce the Latent Linear Adjustment autoencoder which enables estimation of the contribution of a coarse-scale atmospheric circulation proxy to daily precipitation at high-resolution and in a spatially coherent manner. 

The schematic below illustrates a standard autoencoder that encodes daily precipitation fields to the latent space _L_ which are subsequently decoded to yield the reconstructions on the right hand side. 

<img src="https://github.com/christinaheinze/climate-ae-refac/raw/master/documentation/linear_latent_ae1.png" width="800">

To allow for the climate applications of interest, we extend a standard VAE by adding a linear component _h_ to the architecture. To predict circulation-induced precipitation fields, the Latent Linear Adjustment autoencoder combines the linear component _h_, which models the relationship between circulation and the latent space of an autocoder, with the autoencoder's nonlinear decoder. This is illustrated in the schematic below - here, the precipitation fields on the right hand side show the predictions of circulation-induced precipitation. 

<p align="center">
<img src="https://github.com/christinaheinze/climate-ae-refac/raw/master/documentation/linear_latent_ae2.png" width="450">
</p>

To allow for the linearity between circulation and the latent space, the model is trained with an additional penalty in the cost function that encourages this linearity. The proposed model hence leverages robustness advantages of linear models as well as the flexibility of deep neural networks.

## Installing dependencies

You need Python 3.8. The dependencies are managed with [``poetry``](https://python-poetry.org/). To create a virtual environment and install them using poetry, run:

```
python -m virtualenv env
source env/bin/activate
pip install poetry
poetry install
```

## Data

We provide a sample data set which is available on Zenodo: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3949748.svg)](https://doi.org/10.5281/zenodo.3949748)

The sample data set is a subset of the CRCM5-LE ([Leduc et al. 2019](https://journals.ametsoc.org/jamc/article/58/4/663/336/The-ClimEx-Project-A-50-Member-Ensemble-of-Climate)) and preprocessed as described in our manuscript (TODO: add link). The original data can be accessed at the [ClimEx data Access page](https://www.climex-project.org/en/data-access).

To correctly load the data, you need to copy the file [``settings.py``](https://github.com/christinaheinze/latent-linear-adjustment-autoencoders/blob/master/python/settings.py) and rename it to ``local_settings.py``. In ``local_settings.py``, specify (a) where the data is located in ``DATA_PATH``, and (b) where the output should be saved in ``OUT_PATH``. 


## Running experiments

The commands to run the experiments are detailed in ``python/run-experiments.sh``. Note that you need to add the path of the [``python``](https://github.com/christinaheinze/latent-linear-adjustment-autoencoders/blob/master/python) directory to your ``PYTHONPATH`` (see ``python/run-experiments.sh``). 

The first step consists of training the Latent Linear Adjustment autoencoder model. From the ``python`` directory run:

```
python3.7 climate_ae/models/ae/main_ae.py
```

By default, the hyperparameters from the file ``python/climate_ae/models/ae/configs/config_dyn_adj_precip.json`` will be used which correspond to the settings needed to reproduce  the precipitation results reported in the manuscript. For temperature, the corresponding hyperparameters are given in ``python/climate_ae/models/ae/configs/config_dyn_adj_temp.json``.

Each trained model is associated with a so-called ``CHECKPOINT_ID`` which is needed to load a trained model. The ``CHECKPOINT_ID`` is returned as the last logging statement when training the autoencoder and it is also saved in the model outputs that are written to ``OUT_PATH``.

After training the autoencoder, the linear model can be refitted non-iteratively (keeping the autoencoder parameter fixed) and a number of evaluation plots are produced with the following command. The ``CHECKPOINT_ID`` from the trained autoencoder needs to be passed here, such that the correct model is loaded.

```
python3.7 climate_ae/models/ae/main_linear.py --checkpoint_id='CHECKPOINT_ID' --precip=1
```

Finally, the weather generator can be trained using the following command, again passing the ``CHECKPOINT_ID`` from the trained autoencoder:

```
python3.7 climate_ae/models/ae/main_generator.py --checkpoint_id='CHECKPOINT_ID' --precip=1
```

### Command-line arguments and further hyperparameters

#### ``main_ae.py``

The following experimental settings are controlled via command line arguments:
* ``config``: Specifies the path to the config file which contains further hyperparameter settings.
* ``penalty_weight``: Weight of the penalty in the loss function that enforces the linearity between circulation and the latent space of the autoencoder. 
* ``local_json_dir_name``: Directory name where metrics and configs are saved.
* ``dim_latent``: Dimension of the latent space of the autoencoder. 
* ``num_fc_layers``: Number of fully connected layers; only relevant when ``architecture`` is set to ``fc``.
* ``num_conv_layers``: Number of convolutional layers; only relevant when ``architecture`` is set to ``convolutional``.
* ``num_residual_layers``: Number of residual layers; only relevant when ``architecture`` is set to ``convolutional``.
* ``learning_rate``: Learning rate for training the autoencoder.
* ``learning_rate_lm``: Learning rate for training the linear model.
* ``batch_size``: Batch size for training. 
* ``dropout_rate``: Dropout rate.
* ``ae_l2_penalty_weight``: Weight of L2 penalty for autoencoder parameters.
* ``ae_type``: Autoencoder type; can be ``variational`` or ``deterministic``.
* ``architecture``: Autoencoder architecture; can be ``convolutional`` or ``fc`` (fully-connected).
* ``anno_indices``: Number of annotations to use. Here, number of SLP (EOF-derived) time series to use as input _X_ to the linear model _h_. 
* ``lm_l2_penalty_weight``: L2 penalty weight for linear model.
* ``num_epochs``: Number of epochs used for training the model.

Further hyperparameters such as filter and kernel sizes can be set in the ``config`` file.

#### ``main_linear.py``
The following experimental settings are controlled via command line arguments:
* ``checkpoint_id``: Specifies the checkpoint ID of the autoencoder model that should be loaded. 
* ``precip``: Flag whether the loaded model was trained for precipitation (otherwise temperature).
* ``save_nc_files``: Flag whether to save nc files with predictions.


#### ``main_generator.py``
The following experimental settings are controlled via command line arguments:
* ``checkpoint_id``: Specifies the checkpoint ID of the autoencoder model that should be loaded. 
* ``precip``: Flag whether the loaded model was trained for precipitation (otherwise temperature).
* ``save_nc_files``: Flag whether to save nc files with (original as well as emulated) predictions.
* ``var_order``: If set to ``0``, a simple block bootstrap is used. Otherwise, a parameteric bootstrap based on a VAR model with the given order. 
* ``block_size``: Block size for block-bootstrap.
* ``n_bts_samples``: Number of bootstrap samples to generate.
* ``n_steps``: Number of steps to forecast if parametric bootstrap is used. If set to ``0``, forecast will be made for the entire dataset size. 

## Pre-trained models

We provide the checkpoints of the models used to produce the results in the manuscript on Zenodo: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3950045.svg)](https://doi.org/10.5281/zenodo.3950045)

The file ``checkpoints.zip`` needs to be extracted into the directory ``OUT_PATH``. 

For precipitation, the ``CHECKPOINT_ID`` is ``nKGagmsKDb_4249785``. For temperature, it is ``LDifH9DdVh_4383207``. Hence, to e.g. refit the linear model non-iteratively and to produce the evaluation plots as above, run the following command: 

```
python3.7 climate_ae/models/ae/main_linear.py --checkpoint_id='nKGagmsKDb_4249785' --precip=1
```

## ETH-internal: Running on Leonhard

### Installing dependencies

Run the following commands from the root directory of the repository to install the requirements: 
```
bsub -Is -R "rusage[mem=9000, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" bash
module load python_gpu/3.7.4
module load eth_proxy
python -m venv env
source env/bin/activate
pip install -r requirements.txt
exit
```

### Data
Follow the above steps as described under "Data".


### Running experiments
From the login node, run the following commands to launch the precipitation and temperature experiments, respectively:
```
source env/bin/activate
cd launch_scripts
sh submit-precip.sh
sh submit-temp.sh
```

## References
* Heinze-Deml, C., Sippel, S., Pendergrass, A. G., Lehner, F., and Meinshausen, N., 2020: Latent Linear Adjustment autoencoders: A novel method for estimating and emulating dynamic precipitation at high resolution. arXiV preprint  TODO: complete ref
* Leduc, M., A. Mailhot, A. Frigon, J. Martel, R. Ludwig, G.B. Brietzke, M. Giguère, F. Brissette, R. Turcotte, M. Braun, and J.
Scinocca, 2019: The ClimEx Project: A 50-Member Ensemble of Climate Change Projections at 12-km Resolution over
Europe and Northeastern North America with the Canadian Regional Climate Model (CRCM5). J. Appl. Meteor.
Climatol., 58, 663–693, https://doi.org/10.1175/JAMC-D-18-0021.1.

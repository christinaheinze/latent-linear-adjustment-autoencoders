import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import random

from absl import logging
logging.set_verbosity(logging.INFO)
from itertools import chain
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

import local_settings
from climate_ae.models import utils

from climate_ae.data_generator.datahandler import input_fn
import climate_ae.models.ae.eval_utils as eval_utils
import climate_ae.models.ae.climate_utils as climate_utils

import climate_ae.models.ae.train as train
from climate_ae.models.ae.train_linear_model import load_data, predict_latents_and_decode


DEBUG = False


def get_forecasts(model_fitted, annos, var_order, n_steps):
    forecasts_ = []
    if n_steps == 0:
        n_steps = annos.shape[0] - var_order
    n_to_predict = (annos.shape[0] - var_order)//n_steps
    for i in range(n_to_predict):
        forecasts_.append(
            model_fitted.forecast(
                y=annos[(i*n_steps):((i*n_steps)+var_order),:], 
                steps=n_steps))
    forecasts_np = np.concatenate(forecasts_)
    residuals = annos[var_order:(var_order+n_to_predict*n_steps),:] - forecasts_np
    return forecasts_np, residuals


def generate_bts_sample_parametric(model, reg, fitted_, resids_, block_size, 
    dim_x, dim_out, precip):
    n_blocks = int(np.floor(fitted_.shape[0]/block_size))
    n_blocks_dict = {}
    for b in range(n_blocks):
        n_blocks_dict[b] = list(range((b*block_size), ((b+1)*block_size)))

    ho_annos_bts = np.zeros([n_blocks*block_size, dim_x])
    for b in range(n_blocks):
        # get a random block
        block_id = random.randint(0, n_blocks-1)
        # get residuals from this random block
        resampled_resid = resids_[n_blocks_dict[block_id],:]
        fitted_to_add = fitted_[(b*block_size):((b+1)*block_size),:]
        # add to fitted values 
        ho_annos_bts[(b*block_size):((b+1)*block_size),:] = (fitted_to_add + 
            resampled_resid)
 
    # get decoded predictions for bootstrap sample
    shape_ = [ho_annos_bts.shape[0], *dim_out[1:]]
    ho_xhatexp_bts = predict_latents_and_decode(model, reg, ho_annos_bts, shape_)

    # if precipitation: transform back to original scale
    if precip:
        ho_xhatexp_bts = ho_xhatexp_bts ** 2
    return ho_xhatexp_bts


def generate_bts_sample_simple(model, reg, annos, block_size, dim_out, 
    precip):
    dim_x = annos.shape[1]
    n_blocks = int(np.floor(annos.shape[0]/block_size))
    n_blocks_dict = {}
    for b in range(n_blocks):
        n_blocks_dict[b] = list(range((b*block_size), ((b+1)*block_size)))

    annos_bts = np.zeros([n_blocks*block_size, dim_x])
    for b in range(n_blocks):
        # get a random block
        block_id = random.randint(0, n_blocks-1) 
        annos_bts[(b*block_size):((b+1)*block_size),:] = \
            annos[n_blocks_dict[block_id],:] 
 
    # get decoded predictions for bootstrap sample
    shape_ = [annos_bts.shape[0], *dim_out[1:]]
    xhatexp_bts = predict_latents_and_decode(model, reg, annos_bts, shape_)

    # if precipitation: transform back to original scale
    if precip:
        xhatexp_bts = xhatexp_bts ** 2
    return xhatexp_bts


def train_linear_model_and_generate(checkpoint_path, n_bts_samples, var_order, 
    block_size, n_steps, precip, save_nc, seed=1):
    # set seed
    np.random.seed(seed)

    # get configs from model
    with open(os.path.join(checkpoint_path, "hparams.pkl"), 'rb') as f:
        config = pickle.load(f)
    config = utils.config_to_namedtuple(config)  
    model, _ = train.get_models(config)

    # input function
    def input_anno(params, mode, repeat, n_repeat=None):
        dataset = input_fn(params=params, mode=mode, repeat=repeat, 
            n_repeat=n_repeat, shuffle=False)
        if len(params.temp_indices) == 0 and len(params.psl_indices) == 0:
            dataset = dataset.map(lambda x:
                {"inputs": x["inputs"],
                "anno": tf.gather(x["anno"], params.anno_indices, axis=1),
                "year": x["year"],
                "month": x["month"], 
                "day": x["day"]
                })
        elif len(params.temp_indices) == 0:
            dataset = dataset.map(lambda x:
                {"inputs": x["inputs"],
                "anno": tf.concat(
                    (tf.gather(x["anno"], params.anno_indices, axis=1),
                    tf.gather(x["psl_mean_ens"], params.psl_indices, axis=1)),
                    axis=1),
                "year": x["year"],
                "month": x["month"], 
                "day": x["day"]
                })
        elif len(params.psl_indices) == 0:
            dataset = dataset.map(lambda x:
                {"inputs": x["inputs"],
                "anno": tf.concat(
                    (tf.gather(x["anno"], params.anno_indices, axis=1),
                    tf.gather(x["temp_mean_ens"], params.temp_indices, axis=1)),
                    axis=1),
                "year": x["year"],
                "month": x["month"], 
                "day": x["day"]
                })
        else:
            dataset = dataset.map(lambda x:
                {"inputs": x["inputs"],
                "anno": tf.concat(
                    (tf.gather(x["anno"], params.anno_indices, axis=1),
                    tf.gather(x["psl_mean_ens"], params.psl_indices, axis=1),
                    tf.gather(x["temp_mean_ens"], params.temp_indices, axis=1)),
                    axis=1),
                "year": x["year"],
                "month": x["month"], 
                "day": x["day"]
                })
        return dataset

    global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, 
        name="global_step")
    train_inputs = input_anno(params=config, mode="train", 
        repeat=False)
    test_inputs = input_anno(params=config, mode="test1", 
        repeat=False)
    holdout_inputs = input_anno(params=config, mode="test2", 
        repeat=False)

    # dummy run - otherwise, the model wouldn't be fully build
    show_inputs = iter(train_inputs)
    _ = model(next(show_inputs)["inputs"])
    
    # restore model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model, global_step=global_step)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_consumed()

    # get training data for linear latent space model
    tr_inputs, _, tr_latents, tr_annos, _, _, _ = load_data(
        train_inputs, model, subset=True, debug=DEBUG)
    # fit linear model
    reg = LinearRegression().fit(tr_annos, tr_latents)

    # get test data
    te_inputs, te_recons, _, te_annos, _, _, _ = load_data(test_inputs, model, 
        debug=DEBUG)
    # predict latents for test set and decode
    te_xhatexp = predict_latents_and_decode(model, reg, te_annos, 
        np.shape(te_inputs))

    # get holdout data
    ho_inputs, ho_recons, _, ho_annos, ho_years, ho_months, ho_days = \
        load_data(holdout_inputs, model, debug=DEBUG)
    # predict latents for holdout set and decode
    ho_xhatexp = predict_latents_and_decode(model, reg, ho_annos, 
        np.shape(ho_inputs))
    
    # fit parametric bootstrap with VAR model if var_order > 0, 
    # otherwise a simple block bootstrap is used
    ho_xhatexp_bts_list = list()
    if var_order > 0:
        df_tr_annos = pd.DataFrame(data=tr_annos)
        var_model = VAR(df_tr_annos)
        model_fitted = var_model.fit(var_order)
        forecasts, residuals = get_forecasts(model_fitted, ho_annos, var_order, 
            n_steps)
        
        for i in range(n_bts_samples):
            ho_xhatexp_bts_tmp = generate_bts_sample_parametric(model, reg, 
                forecasts, residuals, block_size, np.shape(ho_annos)[1], 
                np.shape(ho_inputs), precip)
            ho_xhatexp_bts_list.append(ho_xhatexp_bts_tmp)
    else:
        # simple block bootstrap
        for i in range(n_bts_samples):
            ho_xhatexp_bts_tmp = generate_bts_sample_simple(model, reg, 
                ho_annos, block_size, np.shape(ho_inputs), precip)
            ho_xhatexp_bts_list.append(ho_xhatexp_bts_tmp)

    ho_inputs_orig = ho_inputs
    tr_inputs_orig = tr_inputs
    te_inputs_orig = te_inputs
    
    if precip:
        # for precipitation: transform back to original scale    
        ho_inputs = ho_inputs ** 2
        ho_recons = ho_recons ** 2
        ho_xhatexp = ho_xhatexp ** 2
        
        te_inputs = te_inputs ** 2
        te_recons = te_recons ** 2
        te_xhatexp = te_xhatexp ** 2

    # setup folder to save results
    current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
    out_dir = os.path.join(checkpoint_path, "eval_{}".format(current_time))
    os.makedirs(out_dir, exist_ok=True)

    # save
    if save_nc:
        climate_utils.save_ncdf_file_high_res_prec(ho_inputs, ho_years, 
            ho_months, ho_days, "ho_input.nc", out_dir)
        climate_utils.save_ncdf_file_high_res_prec(ho_xhatexp, ho_years, 
            ho_months, ho_days, "ho_pred.nc", out_dir)
        for i in range(n_bts_samples):
            climate_utils.save_ncdf_file_high_res_prec(ho_xhatexp_bts_list[i], 
                ho_years[var_order:(var_order+ho_xhatexp_bts_list[i].shape[0])], 
                ho_months[var_order:(var_order+ho_xhatexp_bts_list[i].shape[0])], 
                ho_days[var_order:(var_order+ho_xhatexp_bts_list[i].shape[0])], 
                "ho_pred_bts_{}.nc".format(i), out_dir)
    

    #################
    # plots based on one bootstrap sample
    #################
    ho_xhatexp_bts = ho_xhatexp_bts_list[0]

    # R2 map
    eval_utils.plot_r2_map(te_inputs, te_recons, te_xhatexp, out_dir, "test") 
    eval_utils.plot_r2_map(ho_inputs, ho_recons, ho_xhatexp, out_dir, "holdout") 
    
    # bootstrap plots
    eval_utils.visualize_boots(ho_xhatexp[var_order:,...], ho_xhatexp_bts, 
        out_dir, "holdout")
    eval_utils.visualize_quiz(ho_xhatexp[var_order:,...], ho_xhatexp_bts, 
        out_dir, "holdout")

    eval_utils.plot_timeseries_boots(ho_xhatexp[var_order:,...], ho_xhatexp_bts, 
        out_dir, "holdout", None)
    eval_utils.plot_timeseries_boots_quiz(ho_xhatexp[var_order:,...], 
        ho_xhatexp_bts, out_dir, "holdout", None)

    # visualize reconstructions and interventions
    eval_utils.visualize(te_inputs_orig, te_annos, model, reg, out_dir, "test", 
        transform_back=precip) 
    eval_utils.visualize(ho_inputs_orig, ho_annos, model, reg, out_dir, "holdout", 
        transform_back=precip)
    eval_utils.visualize(tr_inputs_orig, tr_annos, model, reg, out_dir, "train", 
        transform_back=precip)

    #################
    # plots based on all bts samples
    #################
    eval_utils.plot_timeseries_boots_list(ho_xhatexp[var_order:,...], 
        ho_xhatexp_bts_list, out_dir, "holdout", None)
    imgs_boots = eval_utils.visualize_boots_list(ho_xhatexp[var_order:,...], 
        ho_xhatexp_bts_list, out_dir, "holdout")
    np.save(os.path.join(out_dir, "boots_list_img_xhatexp.npy"), imgs_boots[0])
    for i in range(len(imgs_boots[1])):
        np.save(os.path.join(out_dir, 
            "boots_list_img_xhatexp_bts{}.npy".format(i)), imgs_boots[1][i])

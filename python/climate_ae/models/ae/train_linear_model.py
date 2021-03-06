import datetime
import json
import numpy as np
import os
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from absl import logging
logging.set_verbosity(logging.INFO)
from sklearn.linear_model import LinearRegression

import local_settings
from climate_ae.models import utils

from climate_ae.data_generator.datahandler import input_fn
import climate_ae.models.ae.eval_utils as eval_utils
import climate_ae.models.ae.climate_utils as climate_utils

import climate_ae.models.ae.train as train

DEBUG = False


def load_data(inputs, model, subset=False, debug=False):
    # get training data for linear latent space model
    for b, features in enumerate(inputs):
        if debug and b % 10 == 0 and b > 0:
            break
        if b % 100 == 0:
            print(b)
        
        input_ = features["inputs"]
        recon_ = model.autoencode(input_, training=False)["output"]
        anno_ = train.get_annotations(features)
        year_ = features["year"]
        month_ = features["month"]
        day_ = features["day"]
        encodings_ = model.mean_encode(input_, training=False)['z'].numpy()
        # encodings_z = encodings['z'].numpy()
        
        if b == 0:
            inputs = input_
            recons = recon_
            latents = encodings_
            annos = anno_
            years = year_
            months = month_
            days = day_
        else:
            latents = np.r_[latents, encodings_]
            annos = np.r_[annos, anno_]
            if subset and b <= 10:
                # just keep a subset in memory
                inputs = np.r_[inputs, input_]       
                recons = np.r_[recons, recon_]
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
            else:
                inputs = np.r_[inputs, input_]       
                recons = np.r_[recons, recon_]
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
    return inputs, recons, latents, annos, years, months, days


def predict_latents_and_decode(model, reg_model, annos, out_shape):
     # predict latents
    latentshat = reg_model.predict(annos)

    # decode predicted latents
    xhatexp = np.zeros(out_shape)
    for i in range(xhatexp.shape[0]):
        xhatexp[i, ...] = model.decode(np.expand_dims(latentshat[i, ...], 
            axis=0), training=False)["output"]
    
    return xhatexp


def train_linear_model(checkpoint_path, precip, save_nc_files):
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
    tr_inputs, _, tr_latents, tr_annos, _, _, _ = load_data(train_inputs, model, 
        subset=True, debug=DEBUG)
    # fit linear model
    reg = LinearRegression().fit(tr_annos, tr_latents)
        
    # get test data
    te_inputs, te_recons, _, te_annos, te_years, te_months, te_days = \
        load_data(test_inputs, model, debug=DEBUG)
    # predict latents for test set and decode
    te_xhatexp = predict_latents_and_decode(model, reg, te_annos, 
        np.shape(te_inputs))

    # get holdout data
    ho_inputs, ho_recons, _, ho_annos, ho_years, ho_months, ho_days = \
        load_data(holdout_inputs, model, debug=DEBUG)
    # predict latents for holdout set and decode
    ho_xhatexp = predict_latents_and_decode(model, reg, ho_annos, 
        np.shape(ho_inputs))

    # setup folder to save results
    current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
    out_dir = os.path.join(checkpoint_path, "eval_{}".format(current_time))
    os.makedirs(out_dir, exist_ok=True)

    # save
    if save_nc_files:
        climate_utils.save_ncdf_file_high_res_prec(te_inputs, te_years, 
            te_months, te_days, "te_input.nc", out_dir)
        climate_utils.save_ncdf_file_high_res_prec(te_xhatexp, te_years, 
            te_months, te_days, "te_pred.nc", out_dir)
        climate_utils.save_ncdf_file_high_res_prec(ho_inputs, ho_years, 
            ho_months, ho_days, "ho_input.nc", out_dir)
        climate_utils.save_ncdf_file_high_res_prec(ho_xhatexp, ho_years, 
            ho_months, ho_days, "ho_pred.nc", out_dir)
    

    #################
    # plots
    #################

    # R2 map
    r2_maps_test = eval_utils.plot_r2_map(te_inputs, te_recons, te_xhatexp, 
        out_dir, "test") 
    np.save(os.path.join(out_dir, "r2map_test_xxhat.npy"), r2_maps_test[0])
    np.save(os.path.join(out_dir, "r2map_test_xxhatexp.npy"), r2_maps_test[1])

    r2_maps_ho = eval_utils.plot_r2_map(ho_inputs, ho_recons, ho_xhatexp, 
        out_dir, "holdout") 
    np.save(os.path.join(out_dir, "r2map_ho_xxhat.npy"), r2_maps_ho[0])
    np.save(os.path.join(out_dir, "r2map_ho_xxhatexp.npy"), r2_maps_ho[1])

    # MSE map 
    mse_map_test = eval_utils.plot_mse_map(te_inputs, te_recons, te_xhatexp, 
        out_dir, "test") 
    np.save(os.path.join(out_dir, "mse_map_test_xxhat.npy"), mse_map_test[0])
    np.save(os.path.join(out_dir, "mse_map_test_xxhatexp.npy"), mse_map_test[1])

    mse_map_ho = eval_utils.plot_mse_map(ho_inputs, ho_recons, ho_xhatexp, 
        out_dir, "holdout") 
    np.save(os.path.join(out_dir, "mse_map_ho_xxhat.npy"), mse_map_ho[0])
    np.save(os.path.join(out_dir, "mse_map_ho_xxhatexp.npy"), mse_map_ho[1])

    # visualize reconstructions and interventions
    imgs_test = eval_utils.visualize(te_inputs, te_annos, model, reg, out_dir, 
        "test")
    np.save(os.path.join(out_dir, "te_x.npy"), imgs_test[0])
    np.save(os.path.join(out_dir, "te_xhat.npy"), imgs_test[1])
    np.save(os.path.join(out_dir, "te_xhatexp.npy"), imgs_test[2])

    imgs_ho = eval_utils.visualize(ho_inputs, ho_annos, model, reg, out_dir, 
        "holdout")
    np.save(os.path.join(out_dir, "ho_x.npy"), imgs_ho[0])
    np.save(os.path.join(out_dir, "ho_xhat.npy"), imgs_ho[1])
    np.save(os.path.join(out_dir, "ho_xhatexp.npy"), imgs_ho[2])

    imgs_tr = eval_utils.visualize(tr_inputs, tr_annos, model, reg, out_dir, 
        "train")
    np.save(os.path.join(out_dir, "tr_x.npy"), imgs_tr[0])
    np.save(os.path.join(out_dir, "tr_xhat.npy"), imgs_tr[1])
    np.save(os.path.join(out_dir, "tr_xhatexp.npy"), imgs_tr[2])

    # if precipitation data, transform back to original scale
    if precip:
        out_dir_orig = "{}_orig".format(out_dir)
        te_inputs_2 = te_inputs ** 2
        te_recons_2 = te_recons ** 2
        te_xhatexp_2 = te_xhatexp ** 2
        r2_maps_test_orig = eval_utils.plot_r2_map(te_inputs_2, te_recons_2, 
            te_xhatexp_2, out_dir_orig, "test_orig") 
        np.save(os.path.join(out_dir_orig, "r2map_test_orig_xxhat.npy"), 
            r2_maps_test_orig[0])
        np.save(os.path.join(out_dir_orig, "r2map_test_orig_xxhatexp.npy"), 
            r2_maps_test_orig[1])
    
        mse_maps_test_orig = eval_utils.plot_mse_map(te_inputs_2, te_recons_2, 
            te_xhatexp_2, out_dir_orig, "test_orig") 
        np.save(os.path.join(out_dir_orig, "mse_map_test_orig_xxhat.npy"), 
            mse_maps_test_orig[0])
        np.save(os.path.join(out_dir_orig, "mse_map_test_orig_xxhatexp.npy"), 
            mse_maps_test_orig[1])

        ho_inputs_2 = ho_inputs ** 2
        ho_recons_2 = ho_recons ** 2
        ho_xhatexp_2 = ho_xhatexp ** 2
        r2_maps_ho_orig = eval_utils.plot_r2_map(ho_inputs_2, ho_recons_2, ho_xhatexp_2, 
            out_dir_orig, "holdout_orig") 
        np.save(os.path.join(out_dir_orig, "r2map_ho_orig_xxhat.npy"), 
            r2_maps_ho_orig[0])
        np.save(os.path.join(out_dir_orig, "r2map_ho_orig_xxhatexp.npy"), 
            r2_maps_ho_orig[1])

        mse_maps_ho_orig = eval_utils.plot_mse_map(ho_inputs_2, ho_recons_2, 
            ho_xhatexp_2, out_dir_orig, "holdout_orig") 
        np.save(os.path.join(out_dir_orig, "mse_map_ho_orig_xxhat.npy"), 
            mse_maps_ho_orig[0])
        np.save(os.path.join(out_dir_orig, "mse_map_ho_orig_xxhatexp.npy"), 
            mse_maps_ho_orig[1])

        # visualize reconstructions and interventions
        imgs_te_orig = eval_utils.visualize(te_inputs, te_annos, model, reg, 
            out_dir_orig, "test_orig", transform_back=True) 
        np.save(os.path.join(out_dir_orig, "te_orig_x.npy"), imgs_te_orig[0])
        np.save(os.path.join(out_dir_orig, "te_orig_xhat.npy"), imgs_te_orig[1])
        np.save(os.path.join(out_dir_orig, "te_orig_xhatexp.npy"), imgs_te_orig[2])

        imgs_ho_orig = eval_utils.visualize(ho_inputs, ho_annos, model, reg, 
            out_dir_orig, "holdout_orig", transform_back=True) 
        np.save(os.path.join(out_dir_orig, "ho_orig_x.npy"), imgs_ho_orig[0])
        np.save(os.path.join(out_dir_orig, "ho_orig_xhat.npy"), imgs_ho_orig[1])
        np.save(os.path.join(out_dir_orig, "ho_orig_xhatexp.npy"), imgs_ho_orig[2])

        imgs_tr_orig = eval_utils.visualize(tr_inputs, tr_annos, model, reg, 
            out_dir_orig, "train_orig", transform_back=True)    
        np.save(os.path.join(out_dir_orig, "tr_orig_x.npy"), imgs_tr_orig[0])
        np.save(os.path.join(out_dir_orig, "tr_orig_xhat.npy"), imgs_tr_orig[1])
        np.save(os.path.join(out_dir_orig, "tr_orig_xhatexp.npy"), imgs_tr_orig[2])
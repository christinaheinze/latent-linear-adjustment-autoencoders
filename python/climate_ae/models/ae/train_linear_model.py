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
process_additional_holdout_members = False

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


def process_holdout(holdout_datasets, model, reg_model, save_nc_files, out_dir):
    results = {}
    for ho in holdout_datasets:
        print(ho)
        result = load_data(holdout_datasets[ho], model, debug=DEBUG)
        ho_inputs, ho_recons, _, ho_annos, ho_years, ho_months, ho_days = result
        
        # predict latents for holdout set and decode
        ho_xhatexp = predict_latents_and_decode(model, reg_model, ho_annos, 
            np.shape(ho_inputs))
        
        results[ho] = ho_inputs, ho_recons, ho_annos, ho_xhatexp

        if save_nc_files:
            # save
            climate_utils.save_ncdf_file_high_res_prec(ho_inputs, ho_years, ho_months, 
                ho_days, "ho_{}_input.nc".format(ho), out_dir)
            climate_utils.save_ncdf_file_high_res_prec(ho_xhatexp, ho_years, ho_months, 
                ho_days, "ho_{}_pred.nc".format(ho), out_dir)
    return results


def holdout_plots(results, model, reg, label, precip, out_dir, out_dir_orig):
    ho_inputs, ho_recons, ho_annos, ho_xhatexp = results
    r2_maps_ho = eval_utils.plot_r2_map(ho_inputs, ho_recons, 
        ho_xhatexp, out_dir, "holdout_{}".format(label)) 
    mse_map_ho = eval_utils.plot_mse_map(ho_inputs, ho_recons, ho_xhatexp, 
        out_dir, "holdout_{}".format(label)) 
    mean_mse_x_xhat = np.mean(mse_map_ho[0])
    mean_mse_x_xhatexp = np.mean(mse_map_ho[1])
    mean_r2_x_xhat = np.mean(r2_maps_ho[0])
    mean_r2_x_xhatexp = np.mean(r2_maps_ho[1])
    eval_utils.visualize(ho_inputs, ho_annos, model, reg, out_dir, 
        "holdout_{}".format(label)) 

    print("\n#### Holdout ensemble: {}".format(label))
    print("Mean MSE(x, xhat): {}".format(mean_mse_x_xhat))
    print("Mean MSE(x, xhatexp): {}".format(mean_mse_x_xhatexp))
    print("Mean R2(x, xhat): {}".format(mean_r2_x_xhat))
    print("Mean R2(x, xhatexp): {}".format(mean_r2_x_xhatexp))
    
    # save metrics again in checkpoint dir
    save_path = os.path.join(out_dir, "metrics_{}.json".format(label))
    metrics = {'mean_mse_x_xhat': mean_mse_x_xhat, 
        'mean_mse_x_xhatexp': mean_mse_x_xhatexp,
        'mean_r2_x_xhat': mean_r2_x_xhat,
        'mean_r2_x_xhatexp': mean_r2_x_xhatexp}

    with open(save_path, 'w') as result_file:
        json.dump(metrics, result_file, sort_keys=True, indent=4)

    if precip: 
        ho_inputs_2 = ho_inputs ** 2
        ho_recons_2 = ho_recons ** 2
        ho_xhatexp_2 = ho_xhatexp ** 2
        r2_maps_ho_orig = eval_utils.plot_r2_map(ho_inputs_2, ho_recons_2, 
            ho_xhatexp_2, out_dir_orig, "holdout_orig_{}".format(label)) 
        mse_map_ho_orig = eval_utils.plot_mse_map(ho_inputs_2, ho_recons_2, 
            ho_xhatexp_2, out_dir_orig, "holdout_orig_{}".format(label)) 
        mean_mse_x_xhat = np.mean(mse_map_ho_orig[0])
        mean_mse_x_xhatexp = np.mean(mse_map_ho_orig[1])
        mean_r2_x_xhat = np.mean(r2_maps_ho_orig[0])
        mean_r2_x_xhatexp = np.mean(r2_maps_ho_orig[1])
        eval_utils.visualize(ho_inputs, ho_annos, model, reg, out_dir_orig, 
            "holdout_orig_{}".format(label), transform_back=True) 
        print("\n# Orig: {}".format(label))
        print("Mean MSE(x, xhat): {}".format(mean_mse_x_xhat))
        print("Mean MSE(x, xhatexp): {}".format(mean_mse_x_xhatexp))
        print("Mean R2(x, xhat): {}".format(mean_r2_x_xhat))
        print("Mean R2(x, xhatexp): {}".format(mean_r2_x_xhatexp))

        # save metrics again in checkpoint dir
        save_path = os.path.join(out_dir_orig, "metrics_orig_{}.json".format(label))
        metrics = {'mean_mse_x_xhat': mean_mse_x_xhat, 
            'mean_mse_x_xhatexp': mean_mse_x_xhatexp,
            'mean_r2_x_xhat': mean_r2_x_xhat,
            'mean_r2_x_xhatexp': mean_r2_x_xhatexp}

        with open(save_path, 'w') as result_file:
            json.dump(metrics, result_file, sort_keys=True, indent=4)


def train_linear_model(checkpoint_path, load_json, results_path, precip, save_nc_files):
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

    # visualize reconstructions and interventions -- random
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

    # visualize reconstructions and interventions -- quantiles
    idx_quantiles_te = eval_utils.get_field_mse_quantile_idx(te_inputs, te_xhatexp)
    idx_quantiles_ho = eval_utils.get_field_mse_quantile_idx(ho_inputs, ho_xhatexp)
    print("Indices test:")
    print(idx_quantiles_te)
    print("Indices ho:")
    print(idx_quantiles_ho)

    imgs_test = eval_utils.visualize(te_inputs, te_annos, model, reg, out_dir, 
        "test_q_mse", random=False, idx=idx_quantiles_te[0])
    np.save(os.path.join(out_dir, "te_x_q_mse.npy"), imgs_test[0])
    np.save(os.path.join(out_dir, "te_xhat_q_mse.npy"), imgs_test[1])
    np.save(os.path.join(out_dir, "te_xhatexp_q_mse.npy"), imgs_test[2])

    imgs_ho = eval_utils.visualize(ho_inputs, ho_annos, model, reg, out_dir, 
        "holdout_q_mse", random=False, idx=idx_quantiles_ho[0])
    np.save(os.path.join(out_dir, "ho_x_q_mse.npy"), imgs_ho[0])
    np.save(os.path.join(out_dir, "ho_xhat_q_mse.npy"), imgs_ho[1])
    np.save(os.path.join(out_dir, "ho_xhatexp_q_mse.npy"), imgs_ho[2])

    imgs_test = eval_utils.visualize(te_inputs, te_annos, model, reg, out_dir, 
        "test_q_r2", random=False, idx=idx_quantiles_te[1])
    np.save(os.path.join(out_dir, "te_x_q_r2.npy"), imgs_test[0])
    np.save(os.path.join(out_dir, "te_xhat_q_r2.npy"), imgs_test[1])
    np.save(os.path.join(out_dir, "te_xhatexp_q_r2.npy"), imgs_test[2])

    imgs_ho = eval_utils.visualize(ho_inputs, ho_annos, model, reg, out_dir, 
        "holdout_q_r2", random=False, idx=idx_quantiles_ho[1])
    np.save(os.path.join(out_dir, "ho_x_q_r2.npy"), imgs_ho[0])
    np.save(os.path.join(out_dir, "ho_xhat_q_r2.npy"), imgs_ho[1])
    np.save(os.path.join(out_dir, "ho_xhatexp_q_r2.npy"), imgs_ho[2])

    #################
    # save summaries of metrics maps 
    #################

    # mean MSE over entire field
    test_metrics = {}
    test_metrics.update({"mean_mse_x_xhat": np.mean(mse_map_test[0])})
    test_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_map_test[1])})
    # mean R2 over entire field
    test_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_test[0])})
    test_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_test[1])})

    # mean MSE over entire field
    ho_metrics = {}
    ho_metrics.update({"mean_mse_x_xhat": np.mean(mse_map_ho[0])})
    ho_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_map_ho[1])})
    # mean R2 over entire field
    ho_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_ho[0])})
    ho_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_ho[1])})

    metrics = {'test': test_metrics, 'ho': ho_metrics}
    
    # print
    print("Metrics:")
    for entry in metrics:
        print(entry)
        print(metrics[entry])

    # save metrics again in checkpoint dir
    save_path = os.path.join(out_dir, "metrics.json")
    with open(save_path, 'w') as result_file:
        json.dump(metrics, result_file, sort_keys=True, indent=4)

    if load_json:    
        # save metrics in json file
        exp_jsons = os.listdir(results_path)
        exp_json = [f for f in exp_jsons if config.id in f][0]
        exp_json_path = os.path.join(results_path, exp_json)
        results = utils.load_json(exp_json_path)
        results[config.id]['linear_model_test'] = test_metrics
        results[config.id]['linear_model_ho'] = ho_metrics
        
        with open(exp_json_path, 'w') as result_file:
            json.dump(results, result_file, sort_keys=True, indent=4)

        

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

        #################
        # save summaries of metrics maps 
        #################

        # mean MSE over entire field
        test_metrics = {}
        test_metrics.update({"mean_mse_x_xhat": np.mean(mse_maps_test_orig[0])})
        test_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_maps_test_orig[1])})
        # mean R2 over entire field
        test_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_test_orig[0])})
        test_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_test_orig[1])})

        # mean MSE over entire field
        ho_metrics = {}
        ho_metrics.update({"mean_mse_x_xhat": np.mean(mse_maps_ho_orig[0])})
        ho_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_maps_ho_orig[1])})
        # mean R2 over entire field
        ho_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_ho_orig[0])})
        ho_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_ho_orig[1])})

        # save metrics again in checkpoint dir
        save_path = os.path.join(out_dir_orig, "metrics_orig.json")
        metrics = {'test': test_metrics, 'ho': ho_metrics}
        
        # print
        print("Metrics:")
        for entry in metrics:
            print(entry)
            print(metrics[entry])

        with open(save_path, 'w') as result_file:
            json.dump(metrics, result_file, sort_keys=True, indent=4)

        if load_json:
            # save metrics in json file
            exp_jsons = os.listdir(results_path)
            exp_json = [f for f in exp_jsons if config.id in f][0]
            exp_json_path = os.path.join(results_path, exp_json)
            results = utils.load_json(exp_json_path)
            results[config.id]['linear_model_test_orig'] = test_metrics
            results[config.id]['linear_model_ho_orig'] = ho_metrics
            
            with open(exp_json_path, 'w') as result_file:
                json.dump(results, result_file, sort_keys=True, indent=4)

        

    if process_additional_holdout_members:
        holdout_names = [#"kbd", "kbf", "kbh", "kbj", 
            # "kbl", "kbn", "kbo", "kbp", "kbr"]
            # "kbt", "kbu", "kbv", "kbw", "kbx", 
            # "kby", "kbz", "kca", "kcb", "kcc", 
            # "kcd", "kce", "kcf", "kcg", "kch", 
            # "kci", "kcj", "kck", "kcl", "kcm", 
            "kcn", "kco", "kcp", "kcq", "kcr", 
            "kcs", "kct", "kcu", "kcv", "kcw", "kcx"]
        holdout_datasets = {}
        for ho in holdout_names:
            holdout_datasets[ho] = input_anno(params=config, 
                mode="test_{}".format(ho), 
                repeat=False)
        
        # process and save predictions for additional holdout datasets
        results = process_holdout(holdout_datasets, model, reg, save_nc_files, out_dir)

        for ho in results:
            holdout_plots(results[ho], model, reg, ho, precip, out_dir, out_dir_orig)

    
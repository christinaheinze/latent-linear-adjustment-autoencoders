import argparse
import copy
import datetime
import json
import os
import pickle
import tensorflow as tf
import time

from absl import logging

import local_settings
import climate_ae.models.ae.train as train
import climate_ae.models.utils as utils
import climate_ae.experiments_utils.experiment_repo as repo
from climate_ae.data_generator.datahandler import input_fn


parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--config', type=str, 
    default="climate_ae/models/ae/configs/config_dyn_adj_precip.json", 
    help='Path to config file.')
parser.add_argument('--penalty_weight', type=float, help='Penalty weight')                    
parser.add_argument('--local_json_dir_name', type=str,
    help='Folder name to save results jsons.')   
parser.add_argument('--dim_latent', type=int, 
    help='Dimensionality of latent space.')
parser.add_argument('--num_fc_layers', type=int,
    help='Number of fully connected layers.')
parser.add_argument('--num_conv_layers', type=int,
    help='Number of convolutional layers.')
parser.add_argument('--num_residual_layers', type=int,
    help='Number of residual layers.')
parser.add_argument('--learning_rate', type=float, 
    help='Learning rate autoencoder.') 
parser.add_argument('--learning_rate_lm', type=float, 
    help='Learning rate linear model.') 
parser.add_argument('--batch_size', type=int, help='Batch size.')
parser.add_argument('--dropout_rate', type=float, help='Dropout rate.')
parser.add_argument('--ae_l2_penalty_weight', type=float,
    help='L2 penalty weight for AE.')     
parser.add_argument('--ae_type', type=str, help='AE type.') 
parser.add_argument('--architecture', type=str, help='AE architecture.')
parser.add_argument('--anno_indices', type=int, 
    help='Number of annotations to use.')
parser.add_argument('--lm_l2_penalty_weight', type=float,
    help='L2 penalty weight for linear model.')
parser.add_argument('--num_epochs', type=int, help='Number of epochs.')


def main():    
    # parse args and get configs
    args = parser.parse_args()
    logging.set_verbosity(logging.INFO)

    # get configs
    config_dict = utils.get_config(args.config)
    config_dict = utils.update_config(config_dict, args)

    # correct filter numbers for resnet module
    config_dict["num_filters_resnet_conv1"] = \
        config_dict["filter_sizes"][config_dict["num_conv_layers"]-1]
    config_dict["num_filters_resnet_conv2"] = \
        config_dict["filter_sizes"][config_dict["num_conv_layers"]-1]

    if not isinstance(config_dict['anno_indices'], list):
        config_dict['anno_indices'] = list(range(config_dict['anno_indices']))
    if not isinstance(config_dict['temp_indices'], list):
        config_dict['temp_indices'] = list(range(config_dict['temp_indices']))
    if not isinstance(config_dict['psl_indices'], list):
        config_dict['psl_indices'] = list(range(config_dict['psl_indices']))

    config_dict_copy = copy.deepcopy(config_dict)
    config = utils.config_to_namedtuple(config_dict)

    # Initialize the repo to save results
    logging.info("==> Creating repo..")
    exp_repo = repo.ExperimentRepo(local_dir_name=config.local_json_dir_name,
        root_dir=local_settings.OUT_PATH)

    # Create new experiment
    exp_id = exp_repo.create_new_experiment(config.dataset, config_dict_copy)
    config_dict_copy["id"] = exp_id

    # Set up model directory
    current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")
    ckpt_dir = os.path.join(local_settings.OUT_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    model_dir = os.path.join(
        ckpt_dir, "ckpt_{}_{}".format(current_time, exp_id))

    # Save hyperparameter settings
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(os.path.join(model_dir, "hparams.json")):
        with open(os.path.join(model_dir, "hparams.json"), 'w') as f:
            json.dump(config_dict_copy, f, indent=2, sort_keys=True)
        with open(os.path.join(model_dir, "hparams.pkl"), 'wb') as f:
            pickle.dump(config_dict_copy, f)

    # Set optimizers
    global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, 
        name="global_step")
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate, config.decay_every, 
        config.decay_base, staircase=True)
    optimizer_ae = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)
    optimizer_lm = tf.keras.optimizers.Adam(config.learning_rate_lm, 
        epsilon=1e-08)

    # Get data
    def input_anno(params, mode, repeat, n_repeat=None):
        dataset = input_fn(params=params, mode=mode, repeat=repeat, 
            n_repeat=n_repeat, shuffle=True)
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

    ds_train = input_anno(config, "train", repeat=False)
    ds_test1 = input_anno(config, "test1", repeat=False)
    ds_test2 = input_anno(config, "test2", repeat=False)

    # Get models
    ae_model, linear_model = train.get_models(config)
    show_inputs = iter(ds_train)
    _ = ae_model(next(show_inputs)["inputs"])
    anno = train.get_annotations(next(show_inputs))
    _ = linear_model(anno)

    # Set up checkpointing
    ckpt_ae = tf.train.Checkpoint(model=ae_model, global_step=global_step)
    manager = tf.train.CheckpointManager(ckpt_ae, max_to_keep=5, 
        directory=model_dir)

    writer = tf.summary.create_file_writer(manager._directory)
    with writer.as_default(), tf.summary.record_if(
        lambda: tf.equal(tf.math.mod(global_step, config.save_summary_steps), 0)):
        for epoch in range(0, config.num_epochs):
            start_time = time.time()
            train.train_one_epoch(ae_model, linear_model, ds_train, 
                optimizer_ae, optimizer_lm, global_step, config, epoch, 
                training=True)
            te_metrics = train.eval_one_epoch(ae_model, linear_model, 
                ds_test1, os.path.join(manager._directory, "eval"), 
                global_step, config, epoch, training=False)
            ho_metrics = train.eval_one_epoch(ae_model, linear_model, 
                ds_test2, os.path.join(manager._directory, "holdout"), 
                global_step, config, epoch, training=False)
            tr_metrics = train.eval_one_epoch(ae_model, linear_model, 
                ds_train, os.path.join(manager._directory, "train"), 
                global_step, config, epoch, training=False)
                        
            manager.save()
            logging.info("epoch: %d, time: %0.2f" % 
                (epoch, time.time() - start_time))

            if epoch == 0:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                utils.copy_source(dir_path, manager._directory)

    # # Mark experiment as completed
    exp_repo.mark_experiment_as_completed(exp_id, 
        tr_kld = tr_metrics['kl_loss'], 
        tr_lin_loss = tr_metrics['linear_loss'],
        tr_reconstruction_loss = tr_metrics['reconstruction_loss'],
        tr_exp_reconstruction_loss = tr_metrics['exp_reconstruction_loss'],
        tr_auto_reconstruction_loss = tr_metrics['auto_reconstruction_loss'],
        tr_penalty = tr_metrics['penalty'],
        te_kld = te_metrics['kl_loss'], 
        te_lin_loss = te_metrics['linear_loss'],
        te_reconstruction_loss = te_metrics['reconstruction_loss'],
        te_exp_reconstruction_loss = te_metrics['exp_reconstruction_loss'],
        te_auto_reconstruction_loss = te_metrics['auto_reconstruction_loss'],
        te_penalty = te_metrics['penalty'],
        ho_kld = ho_metrics['kl_loss'], 
        ho_lin_loss = ho_metrics['linear_loss'],
        ho_reconstruction_loss = ho_metrics['reconstruction_loss'],
        ho_exp_reconstruction_loss = ho_metrics['exp_reconstruction_loss'],
        ho_auto_reconstruction_loss = ho_metrics['auto_reconstruction_loss'],
        ho_penalty = ho_metrics['penalty'])

    logging.info('CHECKPOINT_ID="{}"'.format(exp_id))


if __name__ == "__main__":
    main()
import numpy as np
import pickle
import tensorflow as tf

from absl import flags, app, logging
from random import shuffle
from tensorflow.python.ops.summary_ops_v2 import histogram, image, scalar

import local_settings

import climate_ae.models.ae.penalties as penalties

from climate_ae.models.ae.module import ConvVarAutoencoderModel
from climate_ae.models.ae.module import ConvDetAutoencoderModel
from climate_ae.models.ae.module import LinearModelPredictLatents
from climate_ae.models.ae.module import FcDetAutoencoderModel
from climate_ae.models.ae.module import FcVarAutoencoderModel



def get_ae_model(config, dim_latent, architecture, ae_type):
    if architecture == "fc" or architecture == "linear":
        if ae_type == "deterministic":
            model = FcDetAutoencoderModel(config, dim_latent)
        elif ae_type == "variational":
            model = FcVarAutoencoderModel(config, dim_latent)
        else:
            raise ValueError 
    elif architecture == "convolutional":
        if ae_type == "deterministic":
            model = ConvDetAutoencoderModel(config, dim_latent)
        elif ae_type == "variational":
            model = ConvVarAutoencoderModel(config, dim_latent)
        else:
            raise ValueError 
    else:
        raise ValueError 

    return model


def get_models(config):
    model = get_ae_model(config, dim_latent=config.dim_latent,
         architecture=config.architecture, ae_type=config.ae_type)
    linear_model = LinearModelPredictLatents(config)

    return model, linear_model


def get_latents(config, model, features, training):
    # get latents
    feats = features["inputs"]
    if config.mean_encode_lm:
        latents = model.mean_encode(feats, training=training)["z"]
    else:
        model_output = model(feats, training=training)
        latents = model_output["z"]
    return latents


def get_annotations(features):
    return tf.cast(features["anno"], tf.float32)


def get_xhatexp(config, model, linear_model, features, training):  
    lhat = get_linear_predictions(linear_model, features)
    return model.decode(lhat, training=False)["output"]


def get_sq_er_x_xhatexp(config, model, linear_model, features, training):
    xhatexp = get_xhatexp(config, model, linear_model, features, training)
    error = features["inputs"] - xhatexp
    sq_er = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])
    return sq_er


def get_linear_predictions(linear_model, features):
    anno = get_annotations(features)
    pred = linear_model(anno)
    return pred


def _loss_fn_lm(latents, linear_model, anno):    
    linear_predictions = linear_model(anno)
    error = latents - linear_predictions
    squared_error = tf.reduce_sum(tf.square(error), axis=1)
    return squared_error


def _loss_fn_ae(model, linear_predictions, inputs, anno, config, training):

    # autoencoder model
    model_output = model(inputs, training=training)

    # reconstruction loss
    reconstruction_loss = tf.reduce_sum(
        tf.square(inputs-model_output["output"]), axis=[1, 2, 3])
        
    # KL loss
    if model.ae_type == "variational":
        kld_z = model_output["kld_z"]
        mean_kld_z = tf.reduce_mean(kld_z)
    else:
        kld_z = 0
        mean_kld_z = 0

    # additional loss term
    penalty = penalties.penalty(model, linear_predictions, inputs, training)

    return reconstruction_loss, mean_kld_z, model_output["output"], penalty


# @tf.function
def _train_step_lm(model, linear_model, features, optimizer_lm, global_step, 
    config, training):

    latents = get_latents(config, model, features, training)
    anno = get_annotations(features)

    with tf.GradientTape() as tape_lm:
        squared_error = _loss_fn_lm(latents, linear_model, anno)

        # mean squared error
        linear_loss = tf.reduce_mean(squared_error)

        # summaries 
        scalar("linear_loss", linear_loss, step=global_step)

        # L2 regularizers
        l2_regularizer_lm = tf.add_n([tf.nn.l2_loss(v) for v in 
            linear_model.trainable_variables if 'bias' not in v.name])
        
        # total loss linear model
        lm_loss = linear_loss + config.lm_l2_penalty_weight*l2_regularizer_lm 
    
    grads = tape_lm.gradient(lm_loss, linear_model.trainable_variables)
    optimizer_lm.apply_gradients(zip(grads, linear_model.trainable_variables))


# @tf.function
def _train_step_ae(model, linear_model, features, optimizer_ae, 
    global_step, config, training, epoch):    
    linear_predictions = get_linear_predictions(linear_model, features)
    anno = get_annotations(features)
    with tf.GradientTape() as tape_ae:
        losses = _loss_fn_ae(model, linear_predictions, features["inputs"], 
            anno, config, training)
        (reconstruction_loss, mean_kld_z, reconstruction, penalty) = losses

        # mean reconstruction loss ae
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        
        # summaries
        scalar("reconstruction_loss", mean_reconstruction_loss, 
            step=global_step)
        scalar("kl_loss", mean_kld_z, step=global_step)
        scalar("penalty", penalty, step=global_step)
        image("images", tf.concat([features["inputs"], reconstruction], axis=2), 
            step=global_step)

        # L2 regularizers
        l2_regularizer_ae = tf.add_n([tf.nn.l2_loss(v) for v in 
            model.trainable_variables if 'bias' not in v.name])
        
        # total loss AE
        ae_loss = (mean_reconstruction_loss + mean_kld_z + 
            config.penalty_weight*penalty + 
            config.ae_l2_penalty_weight*l2_regularizer_ae)
        
    grads = tape_ae.gradient(ae_loss, model.trainable_variables)
    optimizer_ae.apply_gradients(zip(grads, model.trainable_variables))

    global_step.assign_add(1)


def train_one_epoch(model, linear_model, train_features, optimizer_ae, 
    optimizer_lm, global_step, config, epoch, training):
    
    for _features in train_features:
        _train_step_ae(model, linear_model, _features, 
            optimizer_ae, global_step, config, training, epoch)
        _train_step_lm(model, linear_model, _features, optimizer_lm, 
            global_step, config, training)
    

def eval_one_epoch(model, linear_model, test_features, summary_directory, 
    global_step, config, epoch, training):
    metr_reconstruction_loss = tf.metrics.Mean("reconstruction_loss")
    metr_auto_reconstruction_loss = tf.metrics.Mean("auto_reconstruction_loss")
    metr_exp_reconstruction_loss = tf.metrics.Mean("exp_reconstruction_loss")
    metr_kl_loss = tf.metrics.Mean("kl_loss")
    metr_linear_loss = tf.metrics.Mean("linear_loss")
    metr_pen = tf.metrics.Mean("penalty")
    reconstruction_losses = []
    linear_losses = []
    images = []
    images2 = []

    
    for _features in test_features:
        # get predictions and latents
        linear_predictions = get_linear_predictions(linear_model, _features)
        anno = get_annotations(_features)
        latents = get_latents(config, model, _features, training)
        (reconstruction_loss, mean_kld_z, reconstruction, 
            penalty) = _loss_fn_ae(model, linear_predictions, 
                _features["inputs"], anno, config, training)

        # mean encoding 
        out_det = model.autoencode(_features["inputs"], training=training)["output"]
        autoencode_rec_loss = tf.reduce_sum(tf.square(
            out_det - _features["inputs"]), axis=[1, 2, 3])

        squared_error = _loss_fn_lm(latents, linear_model, anno)       

        reconstruction_losses.append(reconstruction_loss.numpy())
        linear_losses.append(squared_error.numpy())
        
        # update mean-metric
        metr_auto_reconstruction_loss(autoencode_rec_loss)
        metr_reconstruction_loss(reconstruction_loss)
        metr_linear_loss(squared_error)
        metr_kl_loss(mean_kld_z)
        metr_pen(penalty)

        se_x_xhatexp = get_sq_er_x_xhatexp(config, model, linear_model, 
            _features, training) 
        exp_reconstruction = get_xhatexp(config, model, linear_model, 
            _features, training) 
        metr_exp_reconstruction_loss(se_x_xhatexp)

        # append input images, and only keep 4 - otherwise we'd get OOM
        images.append(tf.concat([_features["inputs"][0:1, :, :, :], 
            reconstruction[0:1, :, :, :],
            exp_reconstruction[0:1, :, :, :]], axis=2))

        # append input images, and only keep 4 - otherwise we'd get OOM
        images2.append(tf.concat([_features["inputs"][0:1, :, :, :], 
            reconstruction[0:1, :, :, :]], axis=2))
        
        shuffle(images)
        images = images[-4:]

        shuffle(images2)
        images2 = images2[-4:]

    writer = tf.summary.create_file_writer(summary_directory)
    with writer.as_default(), tf.summary.record_if(True):

        scalar("reconstruction_loss", metr_reconstruction_loss.result(), 
            step=global_step)
        scalar("exp_reconstruction_loss", metr_exp_reconstruction_loss.result(), 
            step=global_step)
        scalar("auto_reconstruction_loss", metr_auto_reconstruction_loss.result(), 
            step=global_step)
        scalar("linear_loss", metr_linear_loss.result(), step=global_step)
        scalar("kl_loss", metr_kl_loss.result(), step=global_step)
        scalar("penalty", metr_pen.result(), step=global_step)
        histogram("distribution_reconstruction_loss", 
            np.concatenate(reconstruction_losses, axis=0).flatten(), 
            step=global_step)
        histogram("distribution_linear_loss", 
            np.concatenate(linear_losses, axis=0).flatten(), step=global_step)
        image("images", tf.concat(images, axis=0), step=global_step)
        image("images2", tf.concat(images2, axis=0), step=global_step)


    out_dict = {"reconstruction_loss": metr_reconstruction_loss.result(),
        "exp_reconstruction_loss": metr_exp_reconstruction_loss.result(),
        "auto_reconstruction_loss": metr_auto_reconstruction_loss.result(),
        "kl_loss": metr_kl_loss.result(),
        "linear_loss": metr_linear_loss.result(),
        "penalty": metr_pen.result()
    }

    return out_dict
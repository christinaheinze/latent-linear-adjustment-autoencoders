import numpy as np

import local_settings
from climate_ae.data_generator import temp_psl, prec_psl


def prec_psl_input_fn(params, mode):
    if mode == "train":
        dataset = prec_psl.train(local_settings.DATA_PATH)
    elif mode == "test1":
        dataset = prec_psl.test1(local_settings.DATA_PATH)
    elif mode == "test2":
        dataset = prec_psl.test2(local_settings.DATA_PATH)

    dataset = dataset.map(lambda x, anno, psl, temp, year, month, day: 
        {"inputs": x, "anno": anno, "psl_mean_ens": psl, "temp_mean_ens": temp, 
        "year": year, "month": month, "day": day})

    return dataset


def temp_psl_input_fn(params, mode):
    if mode == "train":
        dataset = temp_psl.train(local_settings.DATA_PATH)
    elif mode == "test1":
        dataset = temp_psl.test1(local_settings.DATA_PATH)
    elif mode == "test2":
        dataset = temp_psl.test2(local_settings.DATA_PATH)

    dataset = dataset.map(lambda x, anno, psl, temp, year, month, day: 
        {"inputs": x, "anno": anno, "psl_mean_ens": psl, "temp_mean_ens": temp, 
        "year": year, "month": month, "day": day})

    return dataset


def input_fn(params, mode, repeat=True, n_repeat=None, shuffle=True, 
    buffer_size=10000, drop_rem=True):
    if params.dataset == 'prec_psl' or params.dataset == 'prec_psl_sds5': # for compatibility with old models
        dataset = prec_psl_input_fn(params, mode)
    elif params.dataset == 'temp_psl' or params.dataset == 'prec_psl_sds6': # for compatibility with old models
        dataset = temp_psl_input_fn(params, mode)
    else:
        raise ValueError("Dataset unknown")

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(params.batch_size, drop_remainder=drop_rem)
    if repeat:
        if n_repeat is None:
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(n_repeat)
    return dataset

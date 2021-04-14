import numpy as np

import local_settings
from climate_ae.data_generator import temp_psl, prec_psl
from climate_ae.data_generator import prec_psl_shorter_training_period as prec_psl_short


def prec_psl_input_fn(params, mode):
    if mode == "train":
        dataset = prec_psl.train(local_settings.DATA_PATH)
    elif mode == "test1":
        dataset = prec_psl.test1(local_settings.DATA_PATH)
    elif mode == "test2":
        dataset = prec_psl.test2(local_settings.DATA_PATH)
    elif mode == "test_kbb":
        dataset = prec_psl.test_kbb(local_settings.DATA_PATH)
    elif mode == "test_kbd":
        dataset = prec_psl.test_kbd(local_settings.DATA_PATH)
    elif mode == "test_kbf":
        dataset = prec_psl.test_kbf(local_settings.DATA_PATH)
    elif mode == "test_kbh":
        dataset = prec_psl.test_kbh(local_settings.DATA_PATH)
    elif mode == "test_kbj":
        dataset = prec_psl.test_kbj(local_settings.DATA_PATH)
    elif mode == "test_kbl":
        dataset = prec_psl.test_kbl(local_settings.DATA_PATH)
    elif mode == "test_kbn":
        dataset = prec_psl.test_kbn(local_settings.DATA_PATH)
    elif mode == "test_kbo":
        dataset = prec_psl.test_kbo(local_settings.DATA_PATH)
    elif mode == "test_kbp":
        dataset = prec_psl.test_kbp(local_settings.DATA_PATH)
    elif mode == "test_kbr":
        dataset = prec_psl.test_kbr(local_settings.DATA_PATH)
    elif mode == "test_kbt":
        dataset = prec_psl.test_kbt(local_settings.DATA_PATH)
    elif mode == "test_kbu":
        dataset = prec_psl.test_kbu(local_settings.DATA_PATH)
    elif mode == "test_kbv":
        dataset = prec_psl.test_kbv(local_settings.DATA_PATH)
    elif mode == "test_kbw":
        dataset = prec_psl.test_kbw(local_settings.DATA_PATH)
    elif mode == "test_kbx":
        dataset = prec_psl.test_kbx(local_settings.DATA_PATH)
    elif mode == "test_kby":
        dataset = prec_psl.test_kby(local_settings.DATA_PATH)
    elif mode == "test_kbz":
        dataset = prec_psl.test_kbz(local_settings.DATA_PATH)
    elif mode == "test_kca":
        dataset = prec_psl.test_kca(local_settings.DATA_PATH)
    elif mode == "test_kcb":
        dataset = prec_psl.test_kcb(local_settings.DATA_PATH)
    elif mode == "test_kcc":
        dataset = prec_psl.test_kcc(local_settings.DATA_PATH)
    elif mode == "test_kcd":
        dataset = prec_psl.test_kcd(local_settings.DATA_PATH)
    elif mode == "test_kce":
        dataset = prec_psl.test_kce(local_settings.DATA_PATH)
    elif mode == "test_kcf":
        dataset = prec_psl.test_kcf(local_settings.DATA_PATH)
    elif mode == "test_kcg":
        dataset = prec_psl.test_kcg(local_settings.DATA_PATH)
    elif mode == "test_kch":
        dataset = prec_psl.test_kch(local_settings.DATA_PATH)
    elif mode == "test_kci":
        dataset = prec_psl.test_kci(local_settings.DATA_PATH)
    elif mode == "test_kcj":
        dataset = prec_psl.test_kcj(local_settings.DATA_PATH)
    elif mode == "test_kck":
        dataset = prec_psl.test_kck(local_settings.DATA_PATH)
    elif mode == "test_kcl":
        dataset = prec_psl.test_kcl(local_settings.DATA_PATH)
    elif mode == "test_kcm":
        dataset = prec_psl.test_kcm(local_settings.DATA_PATH)
    elif mode == "test_kcn":
        dataset = prec_psl.test_kcn(local_settings.DATA_PATH)
    elif mode == "test_kco":
        dataset = prec_psl.test_kco(local_settings.DATA_PATH)
    elif mode == "test_kcp":
        dataset = prec_psl.test_kcp(local_settings.DATA_PATH)
    elif mode == "test_kcq":
        dataset = prec_psl.test_kcq(local_settings.DATA_PATH)
    elif mode == "test_kcr":
        dataset = prec_psl.test_kcr(local_settings.DATA_PATH)
    elif mode == "test_kcs":
        dataset = prec_psl.test_kcs(local_settings.DATA_PATH)
    elif mode == "test_kct":
        dataset = prec_psl.test_kct(local_settings.DATA_PATH)
    elif mode == "test_kcu":
        dataset = prec_psl.test_kcu(local_settings.DATA_PATH)
    elif mode == "test_kcv":
        dataset = prec_psl.test_kcv(local_settings.DATA_PATH)
    elif mode == "test_kcw":
        dataset = prec_psl.test_kcw(local_settings.DATA_PATH)
    elif mode == "test_kcx":
        dataset = prec_psl.test_kcx(local_settings.DATA_PATH)

    dataset = dataset.map(lambda x, anno, psl, temp, year, month, day: 
        {"inputs": x, "anno": anno, "psl_mean_ens": psl, "temp_mean_ens": temp, 
        "year": year, "month": month, "day": day})

    return dataset


def prec_psl_short_input_fn(params, mode):
    if mode == "train":
        dataset = prec_psl_short.train(local_settings.DATA_PATH)
    elif mode == "test1":
        dataset = prec_psl_short.test1(local_settings.DATA_PATH)
    elif mode == "test2":
        dataset = prec_psl_short.test2(local_settings.DATA_PATH)
    elif mode == "test_kbb":
        dataset = prec_psl_short.test_kbb(local_settings.DATA_PATH)
    elif mode == "test_kbd":
        dataset = prec_psl_short.test_kbd(local_settings.DATA_PATH)
    elif mode == "test_kbf":
        dataset = prec_psl_short.test_kbf(local_settings.DATA_PATH)
    elif mode == "test_kbh":
        dataset = prec_psl_short.test_kbh(local_settings.DATA_PATH)
    elif mode == "test_kbj":
        dataset = prec_psl_short.test_kbj(local_settings.DATA_PATH)
    elif mode == "test_kbl":
        dataset = prec_psl_short.test_kbl(local_settings.DATA_PATH)
    elif mode == "test_kbn":
        dataset = prec_psl_short.test_kbn(local_settings.DATA_PATH)
    elif mode == "test_kbo":
        dataset = prec_psl_short.test_kbo(local_settings.DATA_PATH)
    elif mode == "test_kbp":
        dataset = prec_psl_short.test_kbp(local_settings.DATA_PATH)
    elif mode == "test_kbr":
        dataset = prec_psl_short.test_kbr(local_settings.DATA_PATH)
    elif mode == "test_kbt":
        dataset = prec_psl_short.test_kbt(local_settings.DATA_PATH)
    elif mode == "test_kbu":
        dataset = prec_psl_short.test_kbu(local_settings.DATA_PATH)
    elif mode == "test_kbv":
        dataset = prec_psl_short.test_kbv(local_settings.DATA_PATH)
    elif mode == "test_kbw":
        dataset = prec_psl_short.test_kbw(local_settings.DATA_PATH)
    elif mode == "test_kbx":
        dataset = prec_psl_short.test_kbx(local_settings.DATA_PATH)
    elif mode == "test_kby":
        dataset = prec_psl_short.test_kby(local_settings.DATA_PATH)
    elif mode == "test_kbz":
        dataset = prec_psl_short.test_kbz(local_settings.DATA_PATH)
    elif mode == "test_kca":
        dataset = prec_psl_short.test_kca(local_settings.DATA_PATH)
    elif mode == "test_kcb":
        dataset = prec_psl_short.test_kcb(local_settings.DATA_PATH)
    elif mode == "test_kcc":
        dataset = prec_psl_short.test_kcc(local_settings.DATA_PATH)
    elif mode == "test_kcd":
        dataset = prec_psl_short.test_kcd(local_settings.DATA_PATH)
    elif mode == "test_kce":
        dataset = prec_psl_short.test_kce(local_settings.DATA_PATH)
    elif mode == "test_kcf":
        dataset = prec_psl_short.test_kcf(local_settings.DATA_PATH)
    elif mode == "test_kcg":
        dataset = prec_psl_short.test_kcg(local_settings.DATA_PATH)
    elif mode == "test_kch":
        dataset = prec_psl_short.test_kch(local_settings.DATA_PATH)
    elif mode == "test_kci":
        dataset = prec_psl_short.test_kci(local_settings.DATA_PATH)
    elif mode == "test_kcj":
        dataset = prec_psl_short.test_kcj(local_settings.DATA_PATH)
    elif mode == "test_kck":
        dataset = prec_psl_short.test_kck(local_settings.DATA_PATH)
    elif mode == "test_kcl":
        dataset = prec_psl_short.test_kcl(local_settings.DATA_PATH)
    elif mode == "test_kcm":
        dataset = prec_psl_short.test_kcm(local_settings.DATA_PATH)
    elif mode == "test_kcn":
        dataset = prec_psl_short.test_kcn(local_settings.DATA_PATH)
    elif mode == "test_kco":
        dataset = prec_psl_short.test_kco(local_settings.DATA_PATH)
    elif mode == "test_kcp":
        dataset = prec_psl_short.test_kcp(local_settings.DATA_PATH)
    elif mode == "test_kcq":
        dataset = prec_psl_short.test_kcq(local_settings.DATA_PATH)
    elif mode == "test_kcr":
        dataset = prec_psl_short.test_kcr(local_settings.DATA_PATH)
    elif mode == "test_kcs":
        dataset = prec_psl_short.test_kcs(local_settings.DATA_PATH)
    elif mode == "test_kct":
        dataset = prec_psl_short.test_kct(local_settings.DATA_PATH)
    elif mode == "test_kcu":
        dataset = prec_psl_short.test_kcu(local_settings.DATA_PATH)
    elif mode == "test_kcv":
        dataset = prec_psl_short.test_kcv(local_settings.DATA_PATH)
    elif mode == "test_kcw":
        dataset = prec_psl_short.test_kcw(local_settings.DATA_PATH)
    elif mode == "test_kcx":
        dataset = prec_psl_short.test_kcx(local_settings.DATA_PATH)

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
    elif params.dataset == 'prec_psl_short':
        dataset = prec_psl_short_input_fn(params, mode)
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

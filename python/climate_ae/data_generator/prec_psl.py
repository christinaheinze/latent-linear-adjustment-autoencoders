import os
import tensorflow as tf

from climate_ae.data_generator import utils

# dimensionality of annotations
DANNO1 = 1000
DANNO2 = 1
DANNO3 = 1
# image dimensions
HEIGHT = 128
WIDTH = 128
DEPTH = 1
# data type
DTYPE = tf.float64


def train(directory):
    tfrecords_filename = 'train_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test1(directory):
    tfrecords_filename = 'test_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test2(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


### additional holdout members

def test_kbb(directory): # same as test2
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcn(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcn.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcp(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcp.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcr(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcr.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcq(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcq.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcs(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcs.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kct(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kct.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcu(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcu.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcv(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcv.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcw(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcw.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test_kcx(directory):
    tfrecords_filename = 'holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcx.tfrecords'
    subdir = '1955_2070/additional_ho'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds




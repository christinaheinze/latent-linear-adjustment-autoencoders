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
    tfrecords_filename = 'train_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2020_detrend_precip'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test1(directory):
    tfrecords_filename = 'test_prec_psl_short_offset25_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2020_detrend_precip'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


def test2(directory):
    tfrecords_filename = 'holdout_prec_psl_short_offset25_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2020_detrend_precip'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds



def test_kbb(directory): # same as test2
    tfrecords_filename = 'holdout_prec_psl_short_offset25_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbb.tfrecords'
    subdir = '1955_2020_detrend_precip'
    fname = os.path.join(directory, subdir, tfrecords_filename)
    ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
        DANNO2, DANNO3, DTYPE)
    return ds


# def test_kbd(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbd.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbf(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbf.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbh(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbh.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbj(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbj.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbl(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbl.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbn(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbn.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbo(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbo.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbp(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbp.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbr(directory): 
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbr.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbt(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbt.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbu(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbu.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbv(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbv.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbw(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbw.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbx(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbx.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kby(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kby.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kbz(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kbz.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kca(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kca.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcb(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcb.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcc(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcc.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcd(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcd.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kce(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kce.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcf(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcf.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcg(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcg.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kch(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kch.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kci(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kci.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcj(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcj.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kck(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kck.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcl(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcl.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcm(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcm.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcn(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcn.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kco(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kco.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcp(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcp.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcr(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcr.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcq(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcq.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcs(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcs.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kct(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kct.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcu(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcu.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcv(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcv.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcw(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcw.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds


# def test_kcx(directory):
#     tfrecords_filename = 'holdout_prec_psl_short_detrend_precip_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcx.tfrecords'
#     subdir = '1955_2020_detrend_precip/additional_ho'
#     fname = os.path.join(directory, subdir, tfrecords_filename)
#     ds = utils.climate_dataset(directory, fname, HEIGHT, WIDTH, DEPTH, DANNO1,
#         DANNO2, DANNO3, DTYPE)
#     return ds




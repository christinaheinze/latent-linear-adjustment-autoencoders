import rpy2.robjects as robjects
import numpy as np
import random

import os
import tensorflow as tf
import argparse


# helper functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float32_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(value):
    return tf.train.Feature(float_list=tf.train.Int64List(value=value))



def main():
    ''' 
    Converts rda (R data format) files to tfrecords files
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='addho_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcn_kcp_kcr_kcq_kcs_year_all_months_1_2_12_npc_psl1000_mean_ens_temp_psl_detrendTRUE_scale_TRUE_test_split_0.8_temp_disjoint_holdout_kcs.rda',
                        help='input data file name')
    parser.add_argument('--file_path', type=str, default='/u/heinzec/data/climate-linear-latent-adjustment-ae/1955_2070/additional_ho',
                        help='path to input files')
    parser.add_argument('--sname', type=str, default='holdout_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_ho_ens_kcs.tfrecords',
                        help='file name for saving')
    parser.add_argument('--save_dir', type=str, default='/u/heinzec/data/climate-linear-latent-adjustment-ae/1955_2070/additional_ho',
                        help='path for saving files')
    parser.add_argument('--dataset_string', type=str, default='ho',
                        help='substring identifying dataset')
    args = parser.parse_args()

    # read R data
    sub = args.dataset_string    
    robjects.r['load'](os.path.join(args.file_path, args.fname))
    Z = np.array(robjects.r['psl_Z_'+sub])
    Z_psl = np.array(robjects.r['psl_ens_mean_eof_'+sub])
    Z_temp = np.array(robjects.r['temp_ens_mean_eof_'+sub])
    dates = np.array(robjects.r['dates_'+sub])
    years_int = [int(d[1:5]) for d in dates]
    months_int = [int(d[6:8]) for d in dates]
    days_int = [int(d[9:11]) for d in dates]

    Z_psl = np.expand_dims(Z_psl, axis=1)
    Z_temp = np.expand_dims(Z_temp, axis=1)
    years_int = np.expand_dims(np.array(years_int), axis=1)
    months_int = np.expand_dims(np.array(months_int), axis=1)
    days_int = np.expand_dims(np.array(days_int), axis=1)

    images = np.array(robjects.r['prec_mat_sqrt_'+sub])

    filename = os.path.join(args.save_dir, args.sname)
    writer = tf.io.TFRecordWriter(filename)

    for i in range(images.shape[2]):
        # image
        img_t = (images[:, :, i])
        img = _bytes_feature(img_t.tostring())
        
        # annotation
        anno = _float32_feature_list(Z[i,:].astype(np.float32))
        anno_psl = _float32_feature_list(Z_psl[i,:].astype(np.float32))
        anno_temp = _float32_feature_list(Z_temp[i,:].astype(np.float32))
        year = _float32_feature_list(years_int[i,:].astype(np.float32))
        month = _float32_feature_list(months_int[i,:].astype(np.float32))
        day = _float32_feature_list(days_int[i,:].astype(np.float32))
       
        example = tf.train.Example(features=
        tf.train.Features(feature={
            'inputs': img,
            'annotations': anno,
            'psl_mean_ens': anno_psl,
            'temp_mean_ens': anno_temp,
            'year': year,
            'month': month,
            'day': day 
        }))

        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    main()

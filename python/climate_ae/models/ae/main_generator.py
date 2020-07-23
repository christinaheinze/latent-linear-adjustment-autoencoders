import os
import numpy as np

from absl import flags, app, logging
logging.set_verbosity(logging.INFO)

import local_settings
from climate_ae.models.ae.train_linear_model_generator import train_linear_model_and_generate


flags.DEFINE_string(name='checkpoint_id', 
    default='ckpt_200223_1341_LDifH9DdVh_4383207/',
    help='checkpoint directory')
flags.DEFINE_integer(name='precip', default=1, 
    help='Flag whether to generate precipitation (otherwise temperature).')
flags.DEFINE_integer(name='save_nc', default=0, 
    help='Flag whether to save nc files with predictions.')
flags.DEFINE_integer(name='var_order', default=0, 
    help='Order of VAR model. If set to 0, simple block bootstrap is used.')
flags.DEFINE_integer(name='block_size', default=50, 
    help='Block size for block-bootstrap.')
flags.DEFINE_integer(name='n_bts_samples', default=6, 
    help='Number of bootstrap samples to generate.')    
flags.DEFINE_integer(name='n_steps', default=0, 
    help='Number of steps to forecast. If set to 0, forecast over entire dataset.')


def main(_):
    # get results and checkpoint paths
    checkpoint_path = os.path.join(local_settings.OUT_PATH, 'checkpoints')
    checkpoint_folders = os.listdir(checkpoint_path)
    checkpoint_folder = [f for f in checkpoint_folders if flags.FLAGS.checkpoint_id in f]
    if len(checkpoint_folder) == 0:
        raise Exception("No matching folder found.")
    elif len(checkpoint_folder) > 1:
        logging.info(checkpoint_folder)
        raise Exception("More than one matching folder found.")
    else:
        checkpoint_folder = checkpoint_folder[0]
        logging.info("Restoring from {}".format(checkpoint_folder))
    checkpoint_dir = os.path.join(checkpoint_path, checkpoint_folder)

    # retrain linear model and weather generator
    train_linear_model_and_generate(checkpoint_dir, flags.FLAGS.n_bts_samples,
        flags.FLAGS.var_order, flags.FLAGS.block_size, flags.FLAGS.n_steps,
        flags.FLAGS.precip, flags.FLAGS.save_nc)


if __name__ == "__main__":
    app.run(main)

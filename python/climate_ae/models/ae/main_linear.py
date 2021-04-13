import os
import numpy as np

from absl import flags, app, logging
logging.set_verbosity(logging.INFO)

import local_settings
from climate_ae.models.ae.train_linear_model import train_linear_model


flags.DEFINE_string(name='checkpoint_id', default='ckpt_190623_0508_eeDLL34fvB_2278395',
    help='checkpoint directory')
flags.DEFINE_integer(name='load_json', default=0, 
    help='Flag whether to save metrics to json file.')
flags.DEFINE_string(name='results_path', default='exp_jsons', 
    help='checkpoint directory')
flags.DEFINE_integer(name='precip', default=0, 
    help='Flag whether handling precipitation (otherwise temperature).')
flags.DEFINE_integer(name='save_nc_files', default=0, 
    help='Flag whether to save nc files.')


def main(_):
    # get results and checkpoint paths
    results_path = os.path.join(local_settings.OUT_PATH, 
        flags.FLAGS.results_path)
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

    # load or retrain linear model and compute metrics and plots
    train_linear_model(checkpoint_dir, flags.FLAGS.load_json, results_path, 
        flags.FLAGS.precip, 
        flags.FLAGS.save_nc_files)


if __name__ == "__main__":
    app.run(main)

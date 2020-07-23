from datetime import datetime, timezone
import getpass
import io
import json
import pathlib
import uuid
import os
import pickle
import hashlib
import os

import numpy as np
import tensorflow as tf

from absl import logging


def gen_short_uuid(num_chars=None):
    num = uuid.uuid4().int
    alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    res = []
    while num > 0:
        num, digit = divmod(num, len(alphabet))
        res.append(alphabet[digit])
    res2 = ''.join(reversed(res))
    if num_chars is None:
        return res2
    else:
        return res2[:num_chars]


class ExperimentRepo:
    def __init__(self, local_dir_name, root_dir):
        self.local_dir_name = local_dir_name
        self.metadata_filepath = (
            pathlib.Path(__file__).parent / root_dir).resolve()
        self.experiments = {}
        self.experiment_id_by_name = {}
        os.makedirs(self.metadata_filepath, exist_ok=True)
        self.uuid_length = 10

    def gen_short_uuid(self):
        new_id = gen_short_uuid(self.uuid_length)
        assert new_id not in self.experiments
        assert new_id not in self.experiment_id_by_name
        return new_id

    def store_experiment_metadata_local(self):
        with open(self.experiment_metadata_filepath_local, 'w') as f:
            json.dump(self.experiments, f, indent=2, sort_keys=True)

    def create_new_experiment(self, dataset, hyperparameters,
                              name='', description='', verbose=True):
        
        new_id = self.gen_short_uuid()
        try:
            lsf_job_id = os.environ["LSB_JOBID"]
        except:
            lsf_job_id = ""
        new_id = new_id + "_" + lsf_job_id
        creation_time = datetime.now(
            timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
        username = getpass.getuser()
        # name enables extra name for experiment ID
        # experiment_id_by_name maps name to ID
        if name != '':
            assert name not in self.experiment_id_by_name
            assert name not in self.experiments
        new_experiment = {'id': new_id,
                          'hyperparameters': hyperparameters,
                          'username': username,
                          'dataset': dataset,
                          'name': name,
                          'description': description,
                          'creation_time': creation_time,
                          'completed': False,
                          'tr_kld': -1.0, 
                          'tr_lin_loss': -1.0, 
                          'tr_reconstruction_loss': -1.0, 
                          'tr_exp_reconstruction_loss': -1.0,
                          'tr_auto_reconstruction_loss': -1.0,
                          'tr_penalty': -1.0,
                          'te_kld': -1.0, 
                          'te_lin_loss': -1.0, 
                          'te_reconstruction_loss': -1.0, 
                          'te_exp_reconstruction_loss': -1.0,
                          'te_auto_reconstruction_loss': -1.0,
                          'te_penalty': -1.0,
                          'ho_kld': -1.0, 
                          'ho_lin_loss': -1.0, 
                          'ho_reconstruction_loss': -1.0, 
                          'ho_exp_reconstruction_loss': -1.0,
                          'ho_auto_reconstruction_loss': -1.0,
                          'ho_penalty': -1.0
                          }
        self.experiments[new_id] = new_experiment
        if name != '':
            self.experiment_id_by_name[name] = new_id
        
        # save experiment metadata in json file
        local_exp_json_folder = os.path.join(self.metadata_filepath,
            self.local_dir_name)
        if not os.path.isdir(local_exp_json_folder):
            os.makedirs(local_exp_json_folder)
        json_filename_local = new_id + "_exp.json"
        self.experiment_metadata_filepath_local = os.path.join(
            local_exp_json_folder, json_filename_local)
        self.store_experiment_metadata_local()
        
        if verbose:
            logging.info(f'Created a new experiment with id "{new_id}"')

        return new_id


    def mark_experiment_as_completed(self, experiment_id, 
            tr_kld, tr_lin_loss, tr_reconstruction_loss, 
            tr_exp_reconstruction_loss, tr_auto_reconstruction_loss, tr_penalty,
            te_kld, te_lin_loss, te_reconstruction_loss, 
            te_exp_reconstruction_loss, te_auto_reconstruction_loss, te_penalty,
            ho_kld, ho_lin_loss, ho_reconstruction_loss, 
            ho_exp_reconstruction_loss, ho_auto_reconstruction_loss, ho_penalty):
        assert experiment_id in self.experiments
        cur_exp = self.experiments[experiment_id]
        cur_exp['tr_kld'] = float(tr_kld)
        cur_exp['tr_lin_loss'] = float(tr_lin_loss)
        cur_exp['tr_reconstruction_loss'] = float(tr_reconstruction_loss)
        cur_exp['tr_exp_reconstruction_loss'] = float(tr_exp_reconstruction_loss)
        cur_exp['tr_auto_reconstruction_loss'] = float(tr_auto_reconstruction_loss)
        cur_exp['tr_penalty'] = float(tr_penalty)
        cur_exp['te_kld'] = float(te_kld)
        cur_exp['te_lin_loss'] = float(te_lin_loss)
        cur_exp['te_reconstruction_loss'] = float(te_reconstruction_loss)
        cur_exp['te_exp_reconstruction_loss'] = float(te_exp_reconstruction_loss)
        cur_exp['te_auto_reconstruction_loss'] = float(te_auto_reconstruction_loss)
        cur_exp['te_penalty'] = float(te_penalty)
        cur_exp['ho_kld'] = float(ho_kld)
        cur_exp['ho_lin_loss'] = float(ho_lin_loss)
        cur_exp['ho_reconstruction_loss'] = float(ho_reconstruction_loss)
        cur_exp['ho_exp_reconstruction_loss'] = float(ho_exp_reconstruction_loss)
        cur_exp['ho_auto_reconstruction_loss'] = float(ho_auto_reconstruction_loss)
        cur_exp['ho_penalty'] = float(ho_penalty)
        cur_exp['completed'] = True
        self.store_experiment_metadata_local()
    
import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = args["modelSavePath"]
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):

        fname = self.args["logFile"]
        logfile = os.path.join(self.experiment_dir, fname)
        log_file = open(logfile, 'w')

        p = OrderedDict()
        archpars = self.args["architecture"]
        hyperpars = self.args["hyperparameters"]
        p['architecture'] = archpars["type"]
        p['lr'] = hyperpars["lr"]
        p['loss_type'] = self.args["lossType"]
        p['max_epochs'] = hyperpars["maxEpochs"]
        p['weight_decay'] = hyperpars["weightDecay"]
        p['batch_size'] = hyperpars["batchSize"]
        p['optimizer'] = self.args["optimizer"]

        optArgs = ["resumeFromCheckpoint","fineTune"]
        for arg in optArgs:
          if arg in self.args.keys():
             p[arg] = self.args[arg]

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

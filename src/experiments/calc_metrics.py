import os

import yaml
import torch

from src.helpers import read_config
from src.metrics import get_all_metrics
from .train import TrainExperiment


class CalcMetricsExperiment(TrainExperiment):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

    #==========Setters==========
    def _set_correlations(self):
        self.correlations = {k: v  for k, v in read_config(
            os.path.join(self.constraints_dir, "correlations")
        ).items() if 
            k.split("_")[0]==self.cfg['data']['dataset'] or \
            k.split("_")[0]==self.cfg['data']['dataset'].split("_")[0] or \
            k=="base"
        }
        print(self.correlations)
        
    def _set_all(self):
        self._set_correlations()


    #==========Run==========
    def run(self):
        for correlation in self.correlations:
            if correlation+".pth" in os.listdir(self.preds_dir):
                labels_preds = torch.load(os.path.join(self.preds_dir, "{}.pth".format(correlation)))
                metrics_fn = os.path.join(self.preds_dir, "{}.yaml".format(correlation))
                with open(metrics_fn, "r") as f:
                    all_metrics = yaml.safe_load(f)
                all_metrics.update({k: v.item() for k, v in get_all_metrics(labels_preds).items()})
                print(all_metrics)
                with open(metrics_fn, 'w') as f:
                    yaml.dump(all_metrics, f)
            else:
                print("{} file not found.".format(correlation))
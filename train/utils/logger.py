import requests
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only


class ClientLogger(LightningLoggerBase):
    def __init__(self, log_url=None, task_id=None, max_epochs=None):
        self.log_url = log_url
        self.task_id = task_id
        self.max_epochs = max_epochs

    @property
    def name(self):
        return "ClientLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"Logging step: {step}")
        metrics['step'] = step
        metrics['task_id'] = self.task_id
        if self.max_epochs is not None:
            metrics['is_finished'] = metrics['epoch'] >= self.max_epochs
        print(f"Logging metrics: {metrics}")
        if self.log_url:
            requests.post(self.log_url, json=metrics)


    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
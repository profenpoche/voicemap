
import optuna
from olympic.callbacks import Callback

class OptunaPruningCallback(Callback):
    """callback to prune unpromising trials

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        self.monitor = monitor
        self._trial = trial

    def _process(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

    def on_epoch_end(self, epoch, logs=None):
        return self._process(epoch, logs)

    def on_validation_end(self, epoch, logs=None):
        return self._process(epoch, logs)
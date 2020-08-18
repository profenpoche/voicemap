import torch
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
from olympic.callbacks import CSVLogger, Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic import fit
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from voicemap.utils import whiten
# from voicemap.callbacks import DefaultCallback, ProgressBarLogger, CallbackList
from voicemap.metrics import NAMED_METRICS
from voicemap.eval import VerificationMetrics
from voicemap.models import ResidualClassifier, BaselineClassifier
from voicemap.datasets.datasets import letter_to_dataset_dict, gather_datasets
from voicemap.callbacks import OptunaPruningCallback
from config import PATH, DATA_PATH

class training_args:

    def __init__(self, 
                model="resnet", 
                model_path=None,
                dim=1, lr=0.001, 
                weight_decay=0.01, 
                momentum=0.9, 
                epochs=30,
                filters=128,
                batch_size=1500,
                n_seconds=3,
                downsampling=4,
                use_standardized=True,
                spectrogram=True,
                precompute_spect=True,
                window_length=0.02,
                window_hop=0.01,
                device='cuda',
                n_t=0,
                T=None,
                n_f=0,
                F=None
                ):
        self.model: str = model
        self.model_path: str = model_path
        self.dim: int = dim
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.momentum: float = momentum
        self.epochs: int = epochs
        self.filters: int = filters
        self.batch_size: int = batch_size
        self.n_seconds: float = n_seconds
        self.downsampling: int = downsampling
        self.use_standardized: bool = use_standardized
        self.spectrogram: bool = spectrogram
        self.precompute_spect: bool = precompute_spect
        self.window_length: float = window_length
        self.window_hop: float = window_hop
        self.device: str = device
        self.n_t: int = n_t
        self.T = T
        self.n_f: int = n_f
        self.F = F

def calculate_in_channels(args:training_args):
    """ Calculate in_channels based on dim, spectrogram and window_length
    """
    if args.spectrogram:
        if args.dim == 1:
            in_channels = int(args.window_length * 16000) // 2 + 1
        elif args.dim == 2:
            in_channels = 1
        else:
            raise RuntimeError
    else:
        in_channels = 1
    return in_channels


def get_param_str(args:training_args, num_samples=None, num_classes=None):
    """ Generate the parameters string (ex: "model=resnet__dim=1...")
    Add num_samples and/or num_classes if they are given
    """
    excluded_args = ['epochs', 'precompute_spect', 'downsampling', 'window_length','window_hop', 'model_path', 'val_datasets_letters', 'device']
    param_dict = {k: v for k, v in vars(args).items() if not k in excluded_args}
    param_dict['n_seconds'] = f"{param_dict['n_seconds']:.2f}"
    param_dict['datasets'] = param_dict.pop('train_datasets_letters')
    if num_samples is not None:
        param_dict.update({'num_samples': num_samples})
    if num_classes is not None:
        param_dict.update({'num_classes': num_classes})
    param_str = '__'.join([f'{k}={str(v)}' for k, v in param_dict.items()])
    return param_str


def gradient_step(model, optimiser, loss_fn, x, y, epoch, **kwargs):
    """Takes a single gradient step.

    TODO: Accumulent gradients for arbitrary effective batch size
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x, y)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred

def get_prepare_batch_function(spectrogram):
    """ Return the appropriate prepare_batch function according to the spectrogram parameter
    """
    if spectrogram:
        def prepare_batch(batch):
            # Normalise inputs
            # Move to GPU and convert targets to int
            x, y = batch
            return x.double().cuda(), y.long().cuda()
    else:
        def prepare_batch(batch):
            # Normalise inputs
            # Move to GPU and convert targets to int
            x, y = batch
            return whiten(x).cuda(), y.long().cuda()
    return prepare_batch

def batch_metrics(model, y_pred, y, metrics, batch_logs):
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs

def load_model(args:training_args, in_channels, num_classes, device='cuda', model_path=None, verbose=True):
    """ Load weights and return the model loaded
    """
    if model_path is not None:
        args.model_path = model_path
    # Load the weights
    state_dict = torch.load(args.model_path)
    # Create the model with those weights
    model = ResidualClassifier(in_channels, args.filters, [2, 2, 2, 2], num_classes, dim=args.dim)
    model.to(device, dtype=torch.double)
    model.load_state_dict(state_dict=state_dict)
    if verbose:
        print(f"Model loaded : '{args.model_path}'")
    return model

def get_model(args:training_args, in_channels, num_classes, device='cuda', model_path=None, verbose=True):
    """ Create or load a model
    """
    if args.model_path:
        model = load_model(args, in_channels, num_classes, device)
    else:
        if args.model == 'resnet':
            model = ResidualClassifier(in_channels, args.filters, [2, 2, 2, 2], num_classes, dim=args.dim)
        elif args.model == 'baseline':
            model = BaselineClassifier(in_channels, args.filters, 256, num_classes, dim=args.dim)
        else:
            raise RuntimeError
        model.to(device, dtype=torch.double)
    return model

def train(args:training_args, train_datasets_letters, val_datasets_letters, monitor='val_loss', test_size=0.1, metrics=['accuracy'], optunaTrial=None, verbose=True):
    # Calculation of some necessary parameters
    in_channels = calculate_in_channels(args)
    device = torch.device('cuda')

    # Datasets loading
    data, val_datasets = gather_datasets(args, train_datasets_letters, val_datasets_letters)
    num_classes = data.num_classes

    # Creation of the parameters string
    param_str = get_param_str(args, num_samples=len(data), num_classes=num_classes)
    if verbose:
        print(f'Total no. speakers = {num_classes}')

    # loading or creation of the model
    model = get_model(args, in_channels, num_classes, device)

    # Split data as train & test
    indices = range(len(data))
    train_indices, test_indices, _, _ = train_test_split(
        indices,
        indices,
        test_size=test_size
    )
    train = torch.utils.data.Subset(data, train_indices)
    val = torch.utils.data.Subset(data, test_indices)
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)

    # Declaration of optimizer and loss function
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Construction of the callbacks
    callbacks = [
        Evaluate(
            DataLoader(
                train,
                num_workers=cpu_count(),
                batch_sampler=BatchSampler(RandomSampler(train, replacement=True, num_samples=25000), args.batch_size, True)
            ),
            prefix='train_'
        ),
        Evaluate(val_loader)
    ]
    # Add the metrics for the val_datasets chosen
    for letter in val_datasets_letters:
        callbacks.append(
            VerificationMetrics(val_datasets[letter_to_dataset_dict[letter]], 
                num_pairs=25000, 
                prefix=letter_to_dataset_dict[letter]+'_val_')
        )
    callbacks.extend([
        ReduceLROnPlateau(monitor=monitor, patience=5, verbose=True, min_delta=0.25),
        ModelCheckpoint(filepath=PATH + f'/models/{param_str}.pt',
                        monitor=monitor, save_best_only=True, verbose=True),
        CSVLogger(PATH + f'/logs/{param_str}.csv', append=True),
    ])
    if optunaTrial is not None:
        callbacks.append(OptunaPruningCallback(trial=optunaTrial, monitor=monitor))

    # Fit the model
    return fit(
        model,
        opt,
        loss_fn,
        epochs=args.epochs,
        dataloader=train_loader,
        prepare_batch=get_prepare_batch_function(args.spectrogram),
        metrics=metrics,
        callbacks=callbacks,
        update_fn=gradient_step
    )
import logging
import warnings
from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass, field

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import numpy as np

import optuna
from optuna.trial import Trial

import torch
from torch import tensor, nn

from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import PredictionOutput
from transformers.utils.logging import disable_progress_bar
from shared.config import settings
from shared.ranking import AbstractRanker, TrainMode

logger = logging.getLogger('trans-rank')
logging.getLogger('urllib3').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
disable_progress_bar()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_MODELS = [
    'prajjwal1/bert-tiny',
    # 'allenai/scibert_scivocab_uncased',
    # 'climatebert/distilroberta-base-climate-f',
    # 'malteos/scincl',
    # 'distilbert-base',
]


def evaluate(
        # expecting 1D array for y_true and y_pred
        y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor,
        threshold: float = 0.5):
    y_pred_binary = np.where(y_pred > threshold, 1, 0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = None

    results = {
        'ROC AUC': roc_auc,
        'F1': f1_score(y_true, y_pred_binary, zero_division=0),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred_binary)
    }
    logger.debug(' | '.join([f'{key}: {score:.1%}' for key, score in results.items()]))
    return results


def evaluate_trainer(predictions: PredictionOutput):
    with torch.no_grad():
        return evaluate(
            y_true=tensor(predictions.label_ids),
            y_pred=torch.softmax(tensor(predictions.predictions), dim=1)[:, 1]
        )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_class_weights: bool | int = field(
        default=False,
        metadata={'help': 'Whether to use class weights in loss function'})
    class_weights: list[float] | np.ndarray | None = field(
        default=None,
        metadata={'help': 'The weights for each class to be passed to the loss function'})
    model_name: str | None = field(
        default=DEFAULT_MODELS[0],
        metadata={'help': 'Name of the huggingface model'})


class CustomTrainer(Trainer):
    args: CustomTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        y_true = inputs.pop('labels')
        outputs = model(**inputs)
        y_pred = self.activation(outputs.logits)

        criterion = self.loss(weight=self.args.class_weights if self.args.use_class_weights else None)
        loss = criterion(y_pred, y_true)

        return (loss, outputs) if return_outputs else loss

    def predict_proba(self, test_dataset: Dataset) -> np.array:
        predictions = self.predict(test_dataset).predictions
        logits = predictions if torch.is_tensor(predictions) else tensor(predictions)
        # return self.activation(logits).numpy()
        return logits.numpy()  # FIXME: does this still work? returning unscaled logits should be better for ranking


def tokenize(texts: list[str], labels: np.ndarray, model: str, cache_dir: Path | None = None) -> Dataset:
    """
    Returns tokenised dataset from texts and labels using the given model name or filepath
    If using direct path, don't forget to use `--tokenizer` postfix: `{/path/to/model}--tokenizer`

    :param texts:
    :param labels:
    :param model:
    :param cache_dir:
    :return:
    """
    dataset = Dataset.from_dict({
        'text': texts,
        'labels': labels,
    })

    tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=512, cache_dir=cache_dir)
    dataset = dataset.map(
        lambda x: tokenizer(
            x['text'],
            padding='max_length',
            truncation=True
        ),
        batched=True
    )
    dataset.set_format('torch')

    return dataset.remove_columns('text')


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    return torch.tensor(labels.shape[0] / (2 * np.unique_counts(labels).counts),
                        device=device, dtype=torch.float)


class TransRanker(AbstractRanker):
    name: str = 'trans-rank'
    DEFAULT_MODELS = DEFAULT_MODELS

    def __init__(self,
                 model_params: dict[str, Any] | None = None,
                 min_batch_size: int = 2, max_batch_size: int = 32,
                 models: list[str] | None = None,
                 tuning_trials: int = 20,
                 test_split: float = 0.1,
                 train_mode: TrainMode = TrainMode.RESET,
                 **kwargs: dict[str, Any]):
        super().__init__(train_mode, **kwargs)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.models = models or self.DEFAULT_MODELS
        self.model_params = model_params or {}
        self.model: CustomTrainer | None = None
        self.tuning_trials = tuning_trials
        self.test_split = test_split

    @classmethod
    def ensure_offline_models(cls, models: list[str] | None = None):
        from huggingface_hub import snapshot_download
        models = models or cls.DEFAULT_MODELS

        for model in models:
            logger.info(f'Downloading model: {model} so it is available offline in {settings.model_data_path}')
            snapshot_download(
                repo_id=model,
                repo_type='model',
                cache_dir=settings.model_data_path,
                force_download=False,
            )

    @property
    def key(self):
        key = (f'{self.name}'
               f'-{self.train_mode}'
               f'-{self.dataset.batch_strategy}')
        if self.tuning:
            key = f'{key}-tuned'
        return f'{key}-{self.get_hash()}'

    def args(self,
             trial: Trial | None = None,
             weights: list[float] | torch.Tensor | None = None,
             best_params: dict[str, Any] | None = None) -> CustomTrainingArguments:
        base = {
            'output_dir': str(settings.model_data_path),
            'optim': 'adamw_torch',
            'save_strategy': 'no',
            'use_class_weights': 1,
            'class_weights': weights,
            'model_name': self.DEFAULT_MODELS[0],
            'learning_rate': 1e-4,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 12,
            'num_train_epochs': 3,
            'weight_decay': 0.1,
        }

        base |= self.model_params

        if best_params:
            base |= best_params

        if trial:
            batch_size = trial.suggest_int('per_device_train_batch_size', self.min_batch_size, self.max_batch_size)
            base |= {
                'model_name': trial.suggest_categorical('model_name', self.models),
                # 'use_class_weights': trial.suggest_categorical('use_class_weights', [0, 1]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 8),
                'weight_decay': trial.suggest_float('weight_decay', 0, 0.3),
            }
        return CustomTrainingArguments(**base)

    def init(self) -> None:
        pass

    def clear(self):
        self.model = None

    def train(self, idxs: list[int] | None = None, clone: bool = False) -> None:
        if not idxs:
            idxs = self.dataset.seen_data.index.tolist()

        y_true = self.dataset.df.loc[idxs]['label']
        class_weights = compute_class_weights(y_true)

        logger.debug(f'Fitting on {y_true.shape[0]:,} samples ({y_true.sum():,} of which included)')

        if self.train_mode == TrainMode.NEW:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.FULL:
            raise NotImplementedError()
        elif self.train_mode == TrainMode.RESET:
            if self.tuning and self.model is not None and clone:
                args = self.model.args
            elif self.tuning:
                study = optuna.create_study(direction='maximize')
                study.optimize(self.objective(idxs), n_trials=self.tuning_trials, n_jobs=1)
                logger.info(f'Best trial: {study.best_trial.params}')
                logger.debug(f'Hyper-parameter-tuning for {self.name} done with best score {study.best_value}')
                args = self.args(best_params=study.best_params, weights=class_weights)
            else:
                args = self.args(weights=class_weights)

            dataset = tokenize(
                texts=self.dataset.df.loc[idxs]['text'],
                labels=y_true,
                model=args.model_name,
                cache_dir=settings.model_data_path,
            )
            self._train(args, dataset)

    def predict(self, idxs: list[int] | None = None, predict_on_all: bool = True) -> np.ndarray:
        if not idxs:
            idxs = (self.dataset.unseen_data if not predict_on_all else self.dataset.df).index

        if len(idxs) == 0:
            return np.array([])

        y_true = self.dataset.df.loc[idxs]['label']
        dataset = tokenize(
            texts=self.dataset.df.loc[idxs]['text'],
            labels=y_true,
            model=self.model.args.model_name,
            cache_dir=settings.model_data_path,
        )

        logger.debug(f'Predicting on {y_true.shape[0]:,} samples ({y_true.sum():,} of which should be included)')
        y_preds = self.model.predict_proba(dataset)
        logger.debug(f'  > Predictions found {(y_preds > 0.5).sum():,} to be included')
        return y_preds[:, 1]

    def _train(self, args: CustomTrainingArguments, dataset: Dataset):
        logger.debug(f'Training fresh transformer model using "{args.model_name}"')
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            cache_dir=settings.model_data_path,
            num_labels=2,
            ignore_mismatched_sizes=True)

        self.model = CustomTrainer(model=model, args=args, train_dataset=dataset)
        result = self.model.train(resume_from_checkpoint=None)

        logger.debug(f'Time: {result.metrics['train_runtime']:.2f}')
        logger.debug(f'Samples/second: {result.metrics['train_samples_per_second']:.2f}')

    def objective(self, idxs: list[int]) -> Callable[[Trial], float]:
        def run_trial(trial: Trial) -> float:
            logger.debug(f'Running tuning trial {trial.number}')
            training_args = self.args(
                trial=trial,
                weights=compute_class_weights(self.dataset.df.loc[idxs]['label']))
            dataset = tokenize(
                texts=self.dataset.df.loc[idxs]['text'],
                labels=self.dataset.df.loc[idxs]['label'],
                model=training_args.model_name,
                cache_dir=settings.model_data_path,
            )
            dataset = dataset.shuffle()
            train_dataset = dataset.take(n=int(0.6 * len(dataset)))
            test_dataset = dataset.skip(n=int(0.6 * len(dataset)))

            self._train(training_args, train_dataset)
            predictions = self.model.predict(test_dataset)
            results = evaluate_trainer(predictions=predictions)

            logger.info(f'Performance: {results}')

            return results['F1']

        return run_trial

    def _get_params(self, preview: bool = True) -> dict[str, Any]:
        base = {
            'model': self.name,
        }
        if preview:
            return base | {
                'hyperparams': self.model_params
            }
        return base | {
            'hyperparams': self.model_params | {
                'class_weights': self.model.args.class_weights.cpu().tolist(),
                'use_class_weights': self.model.args.use_class_weights,
                'learning_rate': self.model.args.learning_rate,
                'per_device_train_batch_size': self.model.args.per_device_train_batch_size,
                'per_device_eval_batch_size': self.model.args.per_device_eval_batch_size,
                'num_train_epochs': self.model.args.num_train_epochs,
                'weight_decay': self.model.args.weight_decay,
                'model_name': self.model.args.model_name,
                'optim': self.model.args.optim,
            }
        }

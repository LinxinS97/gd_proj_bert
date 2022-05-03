import logging
from typing import Any, Optional, Union, Callable, List

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import tensorboardX
import sklearn.metrics as cls_metric
from torch.cuda.amp import autocast
from tqdm.auto import trange
from transformers import AutoTokenizer

from . import calc_prior
from .loss.effective_number import CB_loss
from .loss.dice_loss import DiceLoss
from .loss.LDAM_loss import LDAMLoss
from .loss.logit_adjustment import logit_adjustment
from .loss.core_loss import CoreLoss
from .. import get_amp_flag
from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import BaseDataset, sample_batch
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


class EndClassifierModel(BaseTorchClassModel):
    def __init__(self,
                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 n_steps: Optional[int] = 10000,
                 loss_type: Optional[str] = 'normal',  # normal: sce, en, dice, ldam
                 score_type: Optional[str] = None,
                 re_sample_type: Optional[str] = None,
                 re_sample_concat: Optional[bool] = False,
                 linear_ratio: Optional[float] = 1,
                 score_threshold: Optional[float] = 0,
                 re_correction: Optional[bool] = False,
                 # softmax: P(A), softmax_distance: P(A)-P(B), new_distance
                 mean_score_type: Optional[str] = None,
                 ndcg_ratio: Optional[float] = 0.2,
                 # Effective number hyper_params
                 beta: Optional[float] = 0.9999,
                 gamma: Optional[float] = 1.0,
                 en_type: Optional[str] = 'softmax',
                 # dice loss hyper_params
                 dice_smooth: Optional[float] = 1e-4,
                 dice_ohem: Optional[float] = 0.0,
                 dice_alpha: Optional[float] = 0.0,
                 dice_square: Optional[bool] = False,
                 # Logit Adjustment
                 adjust_logit: Optional[bool] = True,
                 tau: Optional[float] = 1.0,
                 # ldam
                 max_m: Optional[float] = 0.5,
                 s: Optional[float] = 1,
                 **kwargs: Any
                 ):
        super().__init__()

        self.hyperparas = {
            'batch_size': batch_size,
            'real_batch_size': real_batch_size,
            'test_batch_size': test_batch_size,
            'grad_norm': grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'n_steps': n_steps,
            'binary_mode': binary_mode,
            'loss_type': loss_type,
            'score_type': score_type,
            'mean_score_type': mean_score_type,
            'ndcg_ratio': ndcg_ratio,
            're_sample_type': re_sample_type,
            're_sample_concat': re_sample_concat,
            'linear_ratio': linear_ratio,
            'score_threshold': score_threshold,
            're_correction': re_correction,
            # Effective Number hyper_params
            'beta': beta,
            'gamma': gamma,
            'en_type': en_type,
            # Dice hyper_params
            'dice_smooth': dice_smooth,
            'dice_ohem': dice_ohem,
            'dice_alpha': dice_alpha,
            'dice_square': dice_square,
            # logit adjustment
            'adjust_logit': adjust_logit,
            'tau': tau,
            # ldam loss hyper_params
            'max_m': max_m,
            's': s
        }
        self.model: Optional[BackBone] = None
        self.avg_ranking: Optional[np.ndarray] = None  # [avg_score, ids, label_is_true]
        self.counter: Optional[np.ndarray] = None  # [[idx], [count]]
        self.dev_mode = kwargs['dev_mode']

        self.btm_class_ids = []
        self.top_class_ids = []

        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            **kwargs
        )
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])

    def _calc_loss(self,
                   loss_type,
                   outputs,
                   target,
                   step=None,
                   n_class=None,
                   samples_per_cls=None,
                   reduction='none',
                   tau=1.0,
                   device=None):

        if self.hyperparas['adjust_logit']:
            outputs = logit_adjustment(outputs,
                                       prior=[x / sum(samples_per_cls) for x in samples_per_cls],
                                       tau=tau,
                                       device=device)
        if loss_type == 'normal':
            return cross_entropy_with_probs(outputs, target, reduction=reduction)
        elif loss_type == 'en':
            return CB_loss(outputs,
                           target,
                           samples_per_cls=samples_per_cls,
                           no_of_classes=n_class,
                           # self.hyperparas['loss_type']
                           loss_type='focal',
                           beta=self.hyperparas['beta'],
                           gamma=self.hyperparas['gamma'],
                           device=device)

        elif loss_type == 'dice':
            loss = DiceLoss(with_logits=True,
                            smooth=self.hyperparas['dice_smooth'],
                            ohem_ratio=self.hyperparas['dice_ohem'],
                            alpha=self.hyperparas['dice_alpha'],
                            square_denominator=self.hyperparas['dice_square'],
                            reduction="mean",
                            index_label_position=True)
            return loss(outputs, target)
        elif loss_type == 'ldam':
            loss = LDAMLoss(
                cls_num_list=samples_per_cls,
                max_m=self.hyperparas['max_m'],
                s=self.hyperparas['s'],
                device=device,
                weight=torch.tensor([sum(samples_per_cls) / x for x in samples_per_cls], dtype=torch.float).to(device)
            )
            return loss(outputs, target.long())
        elif loss_type == 'core':
            loss = CoreLoss(
                n_class=n_class,
                noise_prior=torch.tensor([x / sum(samples_per_cls) for x in samples_per_cls]),
            )
            # return loss_cores(step, outputs, target.long(),
            #                   noise_prior=torch.tensor([x / sum(samples_per_cls) for x in samples_per_cls]))
            return loss(outputs, target.long(), step)

    def _calc_score(self, data, y_train):
        # dataloader = self._init_valid_dataloader(data)
        # device = self.model.get_device()
        # self.model.eval()
        ids = np.arange(len(data))
        # for batch in dataloader:
        #     output = self.model(batch)
        #     pred = F.softmax(output, dim=-1)
        #     target = y_train[batch['ids']]
        #     pred_labels = torch.cat((pred_labels, torch.max(pred, dim=1).values))
        #     if self.hyperparas['score_type'] == 'softmax':
        #         p_true = pred[list(range(pred.shape[0])), target]
        #         scores.append(p_true)
        #     elif self.hyperparas['score_type'] == 'softmax_distance':
        #         p_true = pred[list(range(pred.shape[0])), target]
        #         pred[list(range(pred.shape[0])), target] = -1
        #         p_max = torch.max(pred, 1).values
        #         scores.append(p_true - p_max)
        #
        # self.model.train()
        with autocast(enabled=get_amp_flag()):
            pred = self.predict_proba(data)
            pred_labels = np.max(pred, axis=1)
            if self.hyperparas['score_type'] == 'softmax':
                scores = pred[list(range(pred.shape[0])), y_train]
            elif self.hyperparas['score_type'] == 'softmax_distance':
                p_true = pred[list(range(pred.shape[0])), y_train]
                pred[list(range(pred.shape[0])), y_train] = -1
                p_max = np.max(pred, axis=1)
                scores = p_true - p_max

        label_is_true = np.array(data.labels) == y_train
        ranking = np.vstack([scores, ids, label_is_true, pred_labels, y_train])
        return ranking

    def _score_eval(self, ranking: np.ndarray, name: str):

        fpr, tpr, thresholds = cls_metric.roc_curve(ranking[2], ranking[0])
        auc = cls_metric.auc(fpr, tpr)
        ndcg = cls_metric.ndcg_score(np.array([ranking[2]]), np.array([ranking[0]]),
                                     k=int(ranking[0].shape[0] * self.hyperparas['ndcg_ratio']))
        # logger.info(f'{name} auc: {auc}, ndcg: {ndcg}')
        return auc, ndcg

    def _LF_re_correction(self, dataset: BaseDataset, ranking: np.ndarray, n: int):
        LFs = np.array(dataset.weak_labels)
        id_matrix = np.zeros((len(dataset), dataset.n_class))
        prior = np.array(calc_prior(ranking[4].tolist(), dataset.n_class))
        sample_ratio = (prior / sum(prior))

        for i in range(LFs.shape[1]):
            LF = LFs[:, i]
            for label in np.unique(LF)[1:]:  # remove -1
                LFs_set_ids = np.argwhere(LF == label).reshape(-1)
                ranking_LF = ranking[:, LFs_set_ids]
                sorted_rank_ids = np.argsort(ranking_LF[0])[::-1]

                # if not self.is_bert:
                #     class_N = int(sample_ratio[label] * ranking_LF.shape[1]) if \
                #         int(sample_ratio[label] * ranking_LF.shape[1]) <= n // LFs.shape[1] else n // LFs.shape[1]
                # else:
                class_N = int(sample_ratio[label] * n)

                ranking_LF_ids = ranking_LF[:, sorted_rank_ids][1, :class_N]
                id_matrix[:, label][ranking_LF_ids.astype(np.int32)] = 1

        non_conflict = np.argwhere(np.sum(id_matrix, axis=1) <= 1).reshape(-1)
        y_corr = np.argwhere(id_matrix[non_conflict] == 1)

        return y_corr

    def _get_new_dataset(self,
                         dataset: BaseDataset,
                         y_train: np.ndarray,
                         n: int,
                         ranking: np.ndarray,
                         counter: np.ndarray):

        hyperparas = self.hyperparas

        LFs_id = []
        # Threshold (score > 0)
        thres = np.where(ranking[0] >= hyperparas['score_threshold'])[0]
        ranking = ranking[:, thres]

        if hyperparas['re_correction']:
            y_reco = self._LF_re_correction(dataset, ranking, n)
            y_train[y_reco[:, 0]] = y_reco[:, 1]
            LFs_id = y_reco[:, 0]

        sorted_rank_ids = np.argsort(ranking[0])[::-1]
        ranking = ranking[:, sorted_rank_ids]

        sorted_rank_ids = ranking[1]
        ranking = ranking.T

        selected_ids = []
        if hyperparas['re_sample_type'] == 'all_top':
            selected_ids = ranking[:, 1][:n].astype(np.int32)

        elif hyperparas['re_sample_type'] == 'class_top':
            for label in range(dataset.n_class):
                class_ids = ranking[ranking[:, 4] == label]
                # logger.info(f'class {label} clear: {len(class_ids[:, 2][class_ids[:, 2] == 1]) / len(class_ids)}')
                class_ids = class_ids[:n // dataset.n_class]
                class_ids = class_ids[:, 1]
                self.top_class_ids = np.concatenate([self.top_class_ids, class_ids])
                selected_ids.append(class_ids)

                if self.dev_mode:
                    class_ids = class_ids.astype(np.int32)
                    class_dataset = dataset.create_subset(class_ids)
                    class_y = y_train[class_ids]
                    temp = np.array(class_dataset.labels) == class_y
                    logger.info(f'selected class{label} clean: {len(temp[temp == True]) / len(class_y)}')
            selected_ids = np.concatenate(selected_ids).astype(np.int32)

        elif hyperparas['re_sample_type'] == 'hybrid':
            selected_ids = sorted_rank_ids[:n // 2]
            selected_ids = np.concatenate([selected_ids, sorted_rank_ids[-1 * (n // 2):]])

        elif hyperparas['re_sample_type'] == 'class_hybrid':
            sample_size = n // 2 // dataset.n_class
            for label in range(dataset.n_class):
                cur_class_ids = ranking[ranking[:, 4] == label]
                # if len(cur_class_ids[:sample_size]) > 2 * sample_size:
                top_class_ids = cur_class_ids[:sample_size][:, 1]
                btm_class_ids = cur_class_ids[-1 * sample_size:][:, 1]
                # else:
                #     top_class_ids = np.random.choice(cur_class_ids[:sample_size][:, 1], sample_size)
                #     btm_class_ids = np.random.choice(cur_class_ids[-1 * sample_size:][:, 1], sample_size)
                self.btm_class_ids = np.concatenate([self.btm_class_ids, btm_class_ids])
                self.top_class_ids = np.concatenate([self.top_class_ids, top_class_ids])
                class_ids = np.concatenate([top_class_ids, btm_class_ids])
                selected_ids.append(class_ids)

                # if self.dev_mode:
                #     class_ids = class_ids.astype(np.int32)
                #     class_dataset = dataset.create_subset(class_ids.astype(np.int32))
                #     class_y = y_train[class_ids]
                #     temp = np.array(class_dataset.labels) == class_y
                #     logger.info(f'selected clear: {len(temp[temp == True]) / len(class_y)}')
            selected_ids = np.concatenate(selected_ids).astype(np.int32)

        selected_ids = list(set(selected_ids) | set(LFs_id))
        if self.dev_mode:
            new_dataset = dataset.create_subset(selected_ids)
            new_y = y_train[selected_ids]
            prior = calc_prior(new_y.tolist(), dataset.n_class)
            temp = np.array(new_dataset.labels) == new_y
            logger.info(f'clean top n: {len(ranking[:, 2][:n][ranking[:, 2][:n] == 1]) / n}')
            logger.info(f'prior: {prior}')
            logger.info(f'selected clean: {len(temp[temp == True]) / len(new_y)}')

        return selected_ids, y_train

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            evaluation_step: Optional[int] = 10,
            distillation: Optional[bool] = True,
            score_step: Optional[int] = None,
            avg_score_step: Optional[int] = None,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 100,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            grid: Optional[bool] = False,
            **kwargs: Any):

        # log_path = "logs"
        # writer_log = tensorboardX.SummaryWriter(log_path)

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)
        hyperparas = self.config.hyperparas
        logger.info(config)

        n_steps = hyperparas['n_steps']
        samples_per_cls = calc_prior(dataset_train.labels, dataset_train.n_class)
        logger.info(f'original prior: {samples_per_cls}')
        samples_per_cls = calc_prior(y_train.tolist(), dataset_train.n_class)
        logger.info(f'prediction prior: {samples_per_cls}')
        label_is_true = np.array(dataset_train.labels) == y_train
        logger.info(f'original clean: {len(label_is_true[label_is_true == True]) / len(dataset_train)}')

        linear_ratio = np.linspace(1, hyperparas['linear_ratio'], num=n_steps // avg_score_step)

        if hyperparas['real_batch_size'] == -1 or \
                hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor(y_train).to(device)

        if sample_weight is None:
            sample_weight = np.ones(len(dataset_train))
        sample_weight = torch.FloatTensor(sample_weight).to(device)

        if self.model is None:
            model = self._init_model(
                dataset=dataset_train,
                n_class=dataset_train.n_class,
                config=config,
                is_bert=self.is_bert
            )
            self.model = model.to(device)

        train_dataloader = self._init_train_dataloader(
            dataset_train,
            n_steps=hyperparas['n_steps'],
            config=config
        )
        train_dataloader = sample_batch(train_dataloader)


        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(self.model, config)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        score_temp = []
        counter = np.concatenate([np.arange(len(dataset_train)), np.zeros(len(dataset_train))])

        with trange(n_steps, desc="[TRAIN] Warmup stage",
                    unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            cnt = 0
            step = 0
            eval_res = {
                'auc': [],
                'ndcg': [],
                'avg_auc': [],
                'avg_ndcg': []
            }

            self.model.train()
            optimizer.zero_grad()
            while step < n_steps:
                batch = next(train_dataloader)
                outputs = self.model(batch)
                batch_idx = batch['ids'].to(device)
                target = y_train[batch_idx]

                if self.hyperparas['loss_type'] == 'core':
                    loss, score = self._calc_loss(self.hyperparas['loss_type'],
                                                  outputs,
                                                  target,
                                                  step=step,
                                                  samples_per_cls=samples_per_cls,
                                                  n_class=dataset_train.n_class,
                                                  reduction='none',
                                                  tau=self.hyperparas['tau'],
                                                  device=device)
                else:
                    loss = self._calc_loss(self.hyperparas['loss_type'],
                                           outputs,
                                           target,
                                           step=step,
                                           samples_per_cls=samples_per_cls,
                                           n_class=dataset_train.n_class,
                                           reduction='none',
                                           tau=self.hyperparas['tau'],
                                           device=device)

                loss = torch.mean(loss * sample_weight[batch_idx])
                loss.backward()
                cnt += 1

                if cnt % accum_steps == 0:
                    # Clip the norm of the gradients.
                    if hyperparas['grad_norm'] > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), hyperparas['grad_norm'])
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    if valid_flag and step % evaluation_step == 0:

                        metric_value, early_stop_flag, info = self._valid_step(step)
                        if early_stop_flag:
                            logger.info(info)
                            break

                        history[step] = {
                            'loss': loss.item(),
                            f'val_{metric}': metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step': self.best_step,
                        }
                        last_step_log.update(history[step])

                    last_step_log['loss'] = loss.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps:
                        break

                # calc ranking score
                if score_step is not None \
                        and not grid \
                        and distillation \
                        and step >= score_step \
                        and step % score_step == 0:
                    # calculate score
                    ranking = self._calc_score(dataset_train, y_train.cpu().numpy().astype(np.int32))

                    # eval_res['auc'].append(auc)
                    # eval_res['ndcg'].append(ndcg)
                    if avg_score_step:
                        score_temp.append(ranking[0])

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(self.model, config)
        if valid_flag:
            self._reset_valid()
            self._valid_step(-1)
        history = {}
        last_step_log = {}

        if distillation:
            with trange(n_steps, desc="[TRAIN] Distillation stage",
                        unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                eval_res = {
                    'auc': [],
                    'ndcg': [],
                    'avg_auc': [],
                    'avg_ndcg': []
                }
                dataset_ids = np.array([])

                self.model.train()
                optimizer.zero_grad()
                ranked_y = y_train
                y_train = y_train.cpu().numpy().astype(np.int32)
                while step < n_steps:
                    batch = next(train_dataloader)
                    outputs = self.model(batch)
                    batch_idx = batch['ids'].to(device)
                    target = ranked_y[batch_idx]

                    if self.hyperparas['loss_type'] == 'core':
                        loss, score = self._calc_loss(self.hyperparas['loss_type'],
                                                      outputs,
                                                      target,
                                                      step=step,
                                                      samples_per_cls=samples_per_cls,
                                                      n_class=dataset_train.n_class,
                                                      reduction='none',
                                                      tau=self.hyperparas['tau'],
                                                      device=device)
                    else:
                        loss = self._calc_loss(self.hyperparas['loss_type'],
                                               outputs,
                                               target,
                                               step=step,
                                               samples_per_cls=samples_per_cls,
                                               n_class=dataset_train.n_class,
                                               reduction='none',
                                               tau=self.hyperparas['tau'],
                                               device=device)

                    loss = torch.mean(loss * sample_weight[batch_idx])
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients.
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        if valid_flag and step % evaluation_step == 0:

                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                # pass
                                logger.info(info)
                                break

                            history[step] = {
                                'loss': loss.item(),
                                f'val_{metric}': metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step': self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

                    # ranking
                    if score_step is not None \
                            and not grid \
                            and step % score_step == 0:
                        # calculate score
                        ranking = self._calc_score(dataset_train, y_train)

                        # eval_res['auc'].append(auc)
                        # eval_res['ndcg'].append(ndcg)
                        if avg_score_step:
                            score_temp.append(ranking[0])

                        if self.hyperparas['mean_score_type'] is not None \
                                and step % avg_score_step == 0:

                            v_score = np.vstack(score_temp)[(-1 * avg_score_step // score_step):]
                            score_temp = []
                            if self.hyperparas['mean_score_type'] == 'mean':
                                avg_score = np.average(v_score, axis=0)
                            elif self.hyperparas['mean_score_type'] == 'sugi_mean':
                                avg_score = np.average(np.log(1 + v_score + v_score ** 2), axis=0)  # taylor expansion

                            # ranking = self._calc_score(dataset_train, y_train, avg_score)
                            ranking[2] = avg_score
                            # ranking[2, :] = np.cumsum(ranking[2, :]) / np.sum(ranking[2, :])

                            # np.save(f'./ranking/ranking_avg_epoch{step % score_step}_warmup{warm_up}.npy', ranking)
                            # eval_res['avg_auc'].append(auc)
                            # eval_res['avg_ndcg'].append(ndcg)
                            N = int(hyperparas['batch_size'] * score_step * linear_ratio[step // avg_score_step])
                            # N = int(len(dataset_train) * 0.5)
                            selected_ids, y_train = self._get_new_dataset(
                                dataset_train,
                                y_train,
                                N, ranking, counter
                            )
                            dataset_ids = np.concatenate([dataset_ids, selected_ids]).astype(np.int32)
                            ids_set = list(set(dataset_ids))
                            new_dataset = dataset_train.create_subset(ids_set)
                            label_is_true = np.array(new_dataset.labels) == y_train[ids_set]
                            if self.dev_mode:
                                logger.info(f'concat prior: {calc_prior(y_train[ids_set].tolist(), new_dataset.n_class)}')
                                logger.info(f'concat clean: {len(label_is_true[label_is_true == True]) / len(label_is_true)}')
                            ranked_y = torch.Tensor(y_train[ids_set]).to(device)
                            train_dataloader = self._init_train_dataloader(new_dataset, n_steps=score_step, config=config)
                            train_dataloader = sample_batch(train_dataloader)
                            if not hyperparas['re_sample_concat']:
                                dataset_ids = np.array([])

                        elif hyperparas['mean_score_type'] is None:
                            N = int(hyperparas['batch_size'] * score_step * linear_ratio[step // avg_score_step])
                            # N = int(len(dataset_train) * 0.5)
                            selected_ids, y_train = self._get_new_dataset(
                                dataset_train,
                                y_train,
                                N, ranking, counter
                            )
                            dataset_ids = np.concatenate([dataset_ids, selected_ids]).astype(np.int32)
                            ids_set = list(set(dataset_ids))
                            new_dataset = dataset_train.create_subset(ids_set)
                            if self.dev_mode:
                                logger.info(calc_prior(new_dataset.labels, new_dataset.n_class))
                            ranked_y = torch.Tensor(y_train[ids_set]).to(device)
                            train_dataloader = self._init_train_dataloader(new_dataset, n_steps=score_step, config=config)
                            train_dataloader = sample_batch(train_dataloader)
                            if not hyperparas['re_sample_concat']:
                                dataset_ids = np.array([])
        self._finalize()
        return history

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from demo_dataset import DECISION_SAMPLE_WEIGHTS, DemoDataset
from imitation_v5 import autoregressive_imitation_loss, tensor_demo_batch
from ppo_v2 import (
    STICKY_MODEL_ARCHITECTURE,
    StickyActorCriticNetwork,
    atomic_checkpoint,
)


def parse_seeds(value):
    result = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, stop = (int(item) for item in part.split('-', 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def arguments():
    parser = argparse.ArgumentParser(
        description='Distill successful DodgeBlock search trajectories.',
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train-seeds', default='1-12')
    parser.add_argument('--validation-seeds', default='13-16')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--validation-batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--learning-rate-end', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--focus-positive-weight', type=float, default=1.0)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0xBC_005)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--checkpoint-dir', default=str(
        Path.home() / 'dodgeblock-bc-v5/checkpoints'
    ))
    parser.add_argument('--initialize-from')
    parser.add_argument('--weights-from')
    parser.add_argument('--no-amp', action='store_true')
    return parser.parse_args()


def aggregate(metrics, weights):
    total = max(1, sum(weights))
    return {
        key: round(sum(item[key] * weight for item, weight in zip(metrics, weights)) / total, 6)
        for key in metrics[0]
    }


def evaluate_dataset(agent, dataset, batch_size, device, autocast, focus_weight):
    agent.eval()
    metrics = []
    weights = []
    seed_metrics = {}
    with torch.inference_mode():
        for batch, seed in dataset.iter_batches(batch_size):
            observation, actions, targets = tensor_demo_batch(batch, device)
            with autocast:
                logits, _value = agent(observation)
                _loss, result = autoregressive_imitation_loss(
                    logits,
                    observation,
                    actions,
                    targets=targets,
                    focus_positive_weight=focus_weight,
                )
            values = {key: float(value) for key, value in result.items()}
            metrics.append(values)
            weights.append(len(actions))
            seed_metrics.setdefault(seed, [[], []])
            seed_metrics[seed][0].append(values)
            seed_metrics[seed][1].append(len(actions))
    return aggregate(metrics, weights), {
        str(seed): aggregate(values, counts)
        for seed, (values, counts) in seed_metrics.items()
    }


def evaluate_batch(agent, batch, batch_size, device, autocast, focus_weight):
    metrics = []
    weights = []
    agent.eval()
    with torch.inference_mode():
        for start in range(0, len(batch['actions']), batch_size):
            sliced = {
                key: values[start:start + batch_size]
                for key, values in batch.items()
            }
            observation, actions, targets = tensor_demo_batch(sliced, device)
            with autocast:
                logits, _value = agent(observation)
                _loss, result = autoregressive_imitation_loss(
                    logits,
                    observation,
                    actions,
                    targets=targets,
                    focus_positive_weight=focus_weight,
                )
            metrics.append({key: float(value) for key, value in result.items()})
            weights.append(len(actions))
    return aggregate(metrics, weights)


def checkpoint_payload(
    agent,
    optimizer,
    epoch,
    samples,
    args,
    train,
    validation,
    best_validation_loss,
    epochs_without_improvement,
):
    return {
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'samples': samples,
        'best_validation_loss': best_validation_loss,
        'epochs_without_improvement': epochs_without_improvement,
        'frames': 0,
        'stage': 'v5-search-trajectory-bc',
        'model_architecture': STICKY_MODEL_ARCHITECTURE,
        'args': vars(args),
        'dataset_contract': {
            'training': train.contract(),
            'validation': validation.contract(),
        },
    }


def main():
    args = arguments()
    if args.initialize_from and args.weights_from:
        raise ValueError('--initialize-from and --weights-from are mutually exclusive')
    train_seeds = parse_seeds(args.train_seeds)
    validation_seeds = parse_seeds(args.validation_seeds)
    overlap = set(train_seeds) & set(validation_seeds)
    if overlap:
        raise ValueError(f'training and validation seeds overlap: {sorted(overlap)}')
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch size must be positive')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    train = DemoDataset(args.dataset, train_seeds)
    validation = DemoDataset(args.dataset, validation_seeds)
    decision_validation = validation.sample(
        min(65_536, validation.frames),
        np.random.default_rng(args.seed ^ 0xD3C1_5105),
        decision_weighted=True,
    )
    agent = StickyActorCriticNetwork().to(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )
    start_epoch = 1
    samples = 0
    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    if args.initialize_from:
        saved = torch.load(args.initialize_from, map_location=device, weights_only=False)
        agent.load_state_dict(saved['agent'])
        if saved.get('stage') == 'v5-search-trajectory-bc' and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer'])
            start_epoch = int(saved.get('epoch', 0)) + 1
            samples = int(saved.get('samples', 0))
            best_validation_loss = float(
                saved.get('best_validation_loss', float('inf'))
            )
            epochs_without_improvement = int(
                saved.get('epochs_without_improvement', 0)
            )
    elif args.weights_from:
        saved = torch.load(args.weights_from, map_location=device, weights_only=False)
        agent.load_state_dict(saved['agent'])

    rng = np.random.default_rng(args.seed)
    steps_per_epoch = int(np.ceil(train.frames / args.batch_size))
    checkpoint_dir = Path(args.checkpoint_dir)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=not args.no_amp and device.type == 'cuda',
    )
    started = time.time()
    stopped_epoch = args.epochs
    print(json.dumps({
        'event': 'start',
        'stage': 'v5-search-trajectory-bc',
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'parameters': sum(parameter.numel() for parameter in agent.parameters()),
        'training_seeds': train_seeds,
        'validation_seeds': validation_seeds,
        'training_frames': train.frames,
        'validation_frames': validation.frames,
        'steps_per_epoch': steps_per_epoch,
        'focus_positive_weight': args.focus_positive_weight,
        'decision_sample_weights': DECISION_SAMPLE_WEIGHTS,
    }), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        progress = (epoch - 1) / max(1, args.epochs - 1)
        learning_rate = args.learning_rate + progress * (
            args.learning_rate_end - args.learning_rate
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        agent.train()
        train_metrics = []
        train_weights = []
        for _step in range(steps_per_epoch):
            batch = train.sample(args.batch_size, rng)
            observation, actions, targets = tensor_demo_batch(batch, device)
            with autocast:
                logits, _value = agent(observation)
                loss, result = autoregressive_imitation_loss(
                    logits,
                    observation,
                    actions,
                    targets=targets,
                    focus_positive_weight=args.focus_positive_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            values = {key: float(value) for key, value in result.items()}
            values['grad_norm'] = float(grad_norm)
            train_metrics.append(values)
            train_weights.append(len(actions))
            samples += len(actions)

        validation_metrics, validation_by_seed = evaluate_dataset(
            agent,
            validation,
            args.validation_batch_size,
            device,
            autocast,
            args.focus_positive_weight,
        )
        decision_validation_metrics = evaluate_batch(
            agent,
            decision_validation,
            args.validation_batch_size,
            device,
            autocast,
            args.focus_positive_weight,
        )
        metrics = {
            'event': 'epoch',
            'epoch': epoch,
            'samples': samples,
            'learning_rate': learning_rate,
            'elapsed_seconds': round(time.time() - started, 1),
            'training': aggregate(train_metrics, train_weights),
            'validation': validation_metrics,
            'decision_validation': decision_validation_metrics,
            'validation_by_seed': validation_by_seed,
        }
        print(json.dumps(metrics), flush=True)
        improved = decision_validation_metrics['loss'] < best_validation_loss
        if improved:
            best_validation_loss = decision_validation_metrics['loss']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = checkpoint_payload(
            agent,
            optimizer,
            epoch,
            samples,
            args,
            train,
            validation,
            best_validation_loss,
            epochs_without_improvement,
        )
        atomic_checkpoint(checkpoint_dir / f'bc-v5-{epoch:04d}.pt', payload)
        if improved:
            best_path = checkpoint_dir / 'best.pt'
            temporary = checkpoint_dir / 'best.tmp'
            torch.save(payload, temporary)
            temporary.replace(best_path)
        if (
            args.early_stop_patience > 0 and
            epochs_without_improvement >= args.early_stop_patience
        ):
            stopped_epoch = epoch
            print(json.dumps({
                'event': 'early_stop',
                'epoch': epoch,
                'best_validation_loss': best_validation_loss,
                'patience': args.early_stop_patience,
            }), flush=True)
            break

    print(json.dumps({
        'event': 'stop',
        'epoch': stopped_epoch,
        'samples': samples,
        'elapsed_seconds': round(time.time() - started, 1),
        'checkpoint': str(checkpoint_dir / 'latest.pt'),
        'best_checkpoint': str(checkpoint_dir / 'best.pt'),
    }), flush=True)


if __name__ == '__main__':
    main()

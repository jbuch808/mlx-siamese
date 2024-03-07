import argparse
from functools import partial
from glob import glob
import math
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
import os
import time
from utils.dataset import get_dataloaders
from utils.loss import ContrastiveLoss
from utils.model import get_siamese_model
from utils.test import val_epoch
from utils.utility import json_save


DATA_PATH = os.getcwd() + '/../data'
BASE_OUTPUT_PATH = 'models/'

TRAIN_PATH = f'{DATA_PATH}/train/'
VAL_PATH = f'{DATA_PATH}/val/'
DATA_PATHS = [TRAIN_PATH, VAL_PATH]
DATA_LABELS = ['train', 'val']

TRANSFORM_OPTIONS = ['none', 'blur', 'all']

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--model_num", type=int, default=1, choices=[1, 3, 4, 6], help="model architecture number")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--margin", type=float, default=1.0, help="contrastive loss margin")
parser.add_argument("--diverge_max", type=int, default=3, help="max number of divergence epochs")
parser.add_argument("--transform_type", choices=TRANSFORM_OPTIONS, default='all', help="training data transformations options")
parser.add_argument("--pretrained_weights", default=None, help="Path to pretrained weights")
parser.add_argument("--cpu", action="store_true", help="use cpu only")
parser.add_argument("--seed", type=int, default=92, help="random seed")


# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/cifar/main.py
def train_epoch(epoch, model, train_loader, optimizer, loss_fn):
    def train_step(model, anchors, contrasts, labels):
        outputs_anchor, outputs_contrast = model(anchors, contrasts)
        loss = loss_fn(outputs_anchor, outputs_contrast, labels)
        return loss

    losses = []
    samples_per_sec = []

    state = [model.state, optimizer.state,  mx.random.state]

    # TODO: currently compiling crashes with error: IndexError: unordered_map::at: key not found
    # @partial(mx.compile, inputs=state, outputs=state)
    def step(anchors, contrasts, labels):
        train_step_fn = nn.value_and_grad(model, train_step)
        loss, grads = train_step_fn(model, anchors, contrasts, labels)
        optimizer.update(model, grads)
        return loss

    model.train(True)
    for batch_counter, (anchors, contrasts, labels) in enumerate(train_loader):
        tic = time.perf_counter()
        loss = step(anchors, contrasts, labels)
        mx.eval(state)
        toc = time.perf_counter()
        loss = loss.item()
        losses.append(loss)
        throughput = anchors.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        if batch_counter % 10 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {loss:.3f}",
                        f"Throughput: {throughput:.2f} images/second",
                    )
                )
            )

    mean_tr_loss = mx.mean(mx.array(losses))
    samples_per_sec = mx.mean(mx.array(samples_per_sec))
    return mean_tr_loss, samples_per_sec, losses


def train(model, train_loader, val_loader, optimizer, loss_fn, args, output_path, results_path):
    tr_min_loss = math.inf
    val_max_acc = 0
    val_prev_acc = 0
    diverge_count = 0
    results = {'ave_train_loss': [],
               'train_throughput': [],
               'ave_val_acc': [],
               'train_loss': [],
               'val_acc': []
               }
    for epoch in range(args['epochs']):
        tr_loss, tr_throughput, tr_losses = train_epoch(epoch, model, train_loader, optimizer, loss_fn)
        val_acc, val_accs = val_epoch(epoch, model, val_loader, args['margin'])
        print(
            " | ".join(
                (
                    f"Epoch: {epoch}",
                    f"avg. Train loss {tr_loss.item():.3f}",
                    f"Train Throughput: {tr_throughput.item():.2f} images/sec",
                    f"avg. Val Acc {val_acc.item() * 100:.2f}%",
                )
            )
        )

        # Save Model and Info
        if tr_loss < tr_min_loss or val_acc > val_max_acc:
            model.save_weights(f'{output_path}{epoch}.npz')
        results['ave_train_loss'].append(tr_loss.item())
        results['train_throughput'].append(tr_throughput.item())
        results['ave_val_acc'].append(val_acc.item())
        results['train_loss'].extend(tr_losses)
        results['val_acc'].extend(val_accs)
        json_save(results_path, results)

        # Check for Divergence
        if val_acc < val_prev_acc:
            diverge_count += 1
            if diverge_count == args['diverge_max']:
                print('Stopping Early')
                return
        else:
            diverge_count = 0
        val_prev_acc = val_acc


def main():
    # Parse Arguments
    args = parser.parse_args()
    args = vars(args)

    # Set Default Device and Random Seed
    if args['cpu']:
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)
    mx.random.seed(args['seed'])

    # Get Output Paths and Save Args
    output_idx = len(glob(BASE_OUTPUT_PATH + '*')) + 1
    output_path = BASE_OUTPUT_PATH + str(output_idx) + '/'
    os.mkdir(output_path)
    model_info_path = f'{output_path}model_info.json'
    results_path = f'{output_path}results.json'
    json_save(model_info_path, args)

    train_loader, val_loader = get_dataloaders(DATA_PATHS, DATA_LABELS, args['transform_type'], batch_size=args['batch_size'])
    model = get_siamese_model(args['model_num'])

    if args['pretrained_weights'] and os.path.isfile(args['pretrained_weights']):
        model.load_weights(args['pretrained_weights'])

    optimizer = optim.Adam(learning_rate=args['lr'])
    loss_fn = ContrastiveLoss()

    # Train Model
    train(model, train_loader, val_loader, optimizer, loss_fn, args, output_path, results_path)


if __name__ == '__main__':
    main()

from glob import glob
import json
import mlx.core as mx
from mlx.utils import tree_flatten
import os
from utils.plots import plot_train_results


def load_model(model, model_path):
    # Load Training Results
    results_path = f'{model_path}/results.json'
    results, run_index = get_model_train_results(results_path)

    # Plot Training Results
    if results:
        plot_train_results(results)

    # Use Model with highest Validation Accuracy if info was logged
    if run_index:
        weights_path = f'{model_path}/{run_index}.npz'
    # Use last Model saved during training
    else:
        files = glob(f'{model_path}/*.npz')
        files_sorted = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not files_sorted:
            print(f'Model weights not found at: {model_path}')
            return None
        weights_path = files_sorted[-1]
    print(f'Weights Path: {weights_path}')

    # Load pretrained Model weights
    model.load_weights(weights_path)

    # Output Model Information
    params, num_params, info = get_model_info(model, f'{model_path}/info.json')
    print(f'Number of parameters: {num_params}')
    print(params)
    if info:
        print(info)

    return model


def get_model_train_results(results_path):
    results = None
    max_index = None
    if os.path.isfile(results_path):
        with open(results_path, 'r') as file:
            results = json.load(file)

        # Get Training run of max ave val acc
        ave_val_acc = results['ave_val_acc']
        max_index = str(max(enumerate(ave_val_acc), key=lambda x: x[1])[0])

    return results, max_index


def get_model_info(model, info_path):
    # Get # of parameters
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))

    # Get Model Parameters
    params = model.parameters

    # Load Model Info
    info = None
    if os.path.isfile(info_path):
        with open(info_path, 'r') as file:
            info = json.load(file)

    return params, num_params, info


def get_support_set(train_loader, test_loader, val_loader, k):
    # Create Complete Support Set for M-Ways K-Shot testing
    support_set = []
    support_classes = []

    tr_support, tr_support_classes = train_loader.get_support(k)
    test_support, test_support_classes = test_loader.get_support(k)
    val_support, val_support_classes = val_loader.get_support(k)

    for support, support_class in zip(tr_support, tr_support_classes):
        support_set.append(support)
        support_classes.append(support_class)

    for support, support_class in zip(test_support, test_support_classes):
        support_set.append(support)
        support_classes.append(support_class)

    for support, support_class in zip(val_support, val_support_classes):
        support_set.append(support)
        support_classes.append(support_class)

    support_set = mx.array(support_set)
    support_classes = support_classes

    return support_set, support_classes


def json_save(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

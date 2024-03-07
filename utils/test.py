from utils.loss import threshold_contrastive_loss
import mlx.core as mx
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from tqdm import tqdm


def eval_model_inference(model, data_loader, support_set, support_classes, margin, k):
    """
    Test M-Ways K-Shots with a batch size of 1 and Nearest Neighbor for selecting Image Class.

    1. Precompute Feature Vectors for Support Set
    2. For each image (anchor) in the data_loader
        a. Calculate feature vector
        b. Calculate the contrastive loss distance between the anchor feature vector and each feature vector in the
           Support Set.
        c. Find the index in the Support Set with the min distance to the anchor feature vector
        d. Extract the corresponding class of the min distance index.
        e. Bookkeeping
    3. Calculate Statistics
    4. Output Results
    """

    # Calculate Feature Vectors for Support Set
    support_feature_vecs = model(support_set)

    correct = 0
    total = 0
    inference_times = []
    y_true = []
    y_pred = []
    loop = tqdm(data_loader)
    loop.set_description(f'Testing Model | K = {k}')
    model.eval()
    for anchors, anchor_classes in loop:
        # Calculate Predicted Class
        tic = time.perf_counter()
        anchor_feature_vecs = model(anchors)
        preds, dists = threshold_contrastive_loss(anchor_feature_vecs, support_feature_vecs, margin)
        min_dist_index = mx.argmin(dists, axis=0)
        pred_class = support_classes[min_dist_index.item()]
        toc = time.perf_counter()

        # Record Stats
        correct += anchor_classes[0] == pred_class
        total += 1
        inference_time = toc - tic
        inference_times.append(inference_time)
        y_true.append(anchor_classes[0])
        y_pred.append(pred_class)

        # Update Progress Bar
        inference_time_str = f'{inference_time:.6f} seconds'
        loop.set_postfix({"Correct": anchor_classes[0] == pred_class ,"Inference Time": inference_time_str})
        loop.update(1)
    loop.close()

    # Calculate Stats
    acc = correct / total * 100
    ave_inference = mx.mean(mx.array(inference_times))
    cm_labels = list(dict.fromkeys(support_classes))
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    print(
        " | ".join(
            (
                f'K = {k}',
                f"Accuracy: {acc:.2f}%",
                f"Average Inference Time: {ave_inference.item():.6f} seconds",
                f"Average Inference Speed: {1 / ave_inference.item():.2f} images/second",
            )
        )
    )

    return acc, ave_inference, cm, cm_labels


# TODO add confusion matrix calc
def eval_model_batches(model, data_loader, support_set, support_classes, margin, k, batch_size):
    # Calculate Feature Vectors for Support Set
    support_feature_vecs = model(support_set)

    # Tile Matrix to create batch_size * k rows to compare every anchor feature with
    support_feature_vecs_batch = mx.array(np.tile(np.array(support_feature_vecs), (batch_size, 1)))

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    num_support = len(support_classes)
    batch_indexs = mx.array(np.arange(0, batch_size * num_support, num_support))
    loop = tqdm(data_loader)
    loop.set_description(f'Testing Model | K = {k}')
    model.eval()
    for anchors, anchor_classes in loop:
        anchor_feature_vecs = model(anchors)
        anchor_feature_vecs_batch = mx.array(np.repeat(np.array(anchor_feature_vecs), num_support, axis=0))
        if len(anchor_feature_vecs_batch) != len(support_feature_vecs_batch):
            support_feature_vecs_batch = mx.array(np.tile(np.array(support_feature_vecs), (len(anchors), 1)))
            batch_indexs = mx.array(np.arange(0, len(anchors) * num_support, num_support))

        preds, dists = threshold_contrastive_loss(anchor_feature_vecs_batch, support_feature_vecs_batch, margin)

        dists = dists.reshape(-1, num_support)
        min_dist_indexs = mx.argmin(dists, axis=1)
        min_dist_indexs += batch_indexs

        batch_support_classes_indexs = min_dist_indexs % num_support
        batch_support_classes = np.array([support_classes[idx.item()] for idx in batch_support_classes_indexs])

        batch_correct = np.sum(anchor_classes == batch_support_classes)
        batch_total = len(anchors)
        batch_acc = batch_correct / batch_total * 100
        correct += batch_correct
        total += batch_total
        y_true.extend(anchor_classes)
        y_pred.extend(batch_support_classes)

        batch_acc_str = f'{batch_acc:.2f}%'
        loop.set_postfix({"Batch Acc": batch_acc_str})
        loop.update(1)

    loop.close()
    acc = correct / total * 100
    print(f'K = {k} | Accuracy: {acc}%')
    return acc, y_true, y_pred


def val_epoch(epoch, model, val_loader, margin):
    """
    Test Model Binary Classification (whether an image pair is the same class or not) on the Validation Set
    """
    model.eval()
    accs = []
    for batch_counter, (anchors, contrasts, labels) in enumerate(val_loader):
        outputs_anchor, outputs_contrast = model(anchors, contrasts)
        preds, _ = threshold_contrastive_loss(outputs_anchor, outputs_contrast, margin)
        acc = (mx.sum(preds == labels) / len(preds)).item()

        accs.append(acc)

        if batch_counter % 10 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Val Acc {acc:.3f}",
                    )
                )
            )

    mean_val_acc = mx.mean(mx.array(accs))
    return mean_val_acc, accs

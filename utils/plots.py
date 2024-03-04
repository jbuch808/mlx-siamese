import matplotlib.pyplot as plt
import seaborn as sns


def plot_train_results(results):
    ave_train_loss = results['ave_train_loss']
    ave_val_acc = results['ave_val_acc']
    train_loss = results['train_loss']

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)
    ax = axs.flat
    ax[0].plot(train_loss)
    ax[0].title.set_text('Train Loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].set_ylim(0, 1)
    ax[1].plot(ave_train_loss)
    ax[1].title.set_text('Average Train Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].grid(True)
    ax[1].set_ylim(0, 1)
    ax[2].plot(ave_val_acc)
    ax[2].title.set_text('Average Validation Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy %')
    ax[2].grid(True)


def plot_confusion(cm, title, labels):
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

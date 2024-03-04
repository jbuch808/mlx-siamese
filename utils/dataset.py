import cv2 as cv
import mlx.core as mx
import numpy as np
import pickle
import random
from utils.transforms import get_transform


def get_dataloaders(data_paths, data_labels, transform_type, k=1, batch_size=32):
    info_path = f'{data_paths[0]}../dataset_info.pickle'
    with open(info_path, 'rb') as handle:
        info = pickle.load(handle)

    output = []
    for data_path, data_label in zip(data_paths, data_labels):
        transform = get_transform(data_label, transform_type)
        ds = AslDataset(data_path, info[data_label], data_label, transform)
        if data_label == 'train' or data_label == 'val':
            output.append(TrainDataLoader(ds, batch_size=batch_size))
        else:
            output.append(EvalDataLoader(ds, k=k, batch_size=batch_size))
    return output


class AslDataset:
    def __init__(self, data_path, data_info, loader_type, transform):
        super().__init__()

        self.image_paths = data_path

        self.class_to_range = data_info['data_ranges']
        self.class_to_class_idx = data_info['class_to_class_idx']
        self.class_idx_to_class = data_info['class_idx_to_class']

        self.num_classes = len(self.class_to_range.keys())
        self.num_per_class = 3000
        self.num_images = self.num_classes * self.num_per_class
        self.loader_type = loader_type
        self.transform = transform

    def __getitem__(self, index):
        # Check if valid index
        if index < 0 or index >= self.num_images:
            return None

        c1 = self.__get_index_class__(index)
        if self.loader_type == 'train' or self.loader_type == 'val':
            # Randomly Choose second image to be the same class or another class
            if random.random() <= 0.5:
                # Select same class
                c2 = c1
                label = True
            else:
                # Select a random other class
                c2 = self.__get_random_class__(self.class_to_class_idx[c1])
                label = False

            # Get Random image from Class of c2
            data_range = self.class_to_range[c2]
            idx2 = random.randint(data_range[0], data_range[1])

            # Load Images
            img1 = self.__get_image__(index)
            img2 = self.__get_image__(idx2)

            return img1, img2, label
        else:
            img1 = self.__get_image__(index)
            return img1, c1

    def __get_image__(self, idx):
        path = f'{self.image_paths}{idx}.jpg'
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        img_t = self.transform(image=img)
        return img_t['image']

    def __get_index_class__(self, idx):
        # Return the Class corresponding to an image's index in the dataset
        # Check if valid index
        if idx < 0 or idx >= self.num_images:
            return None

        return self.class_idx_to_class[idx // self.num_per_class]

    def __get_random_class__(self, exclude):
        # Loop until randomly generated class is not equal to exclude.
        while True:
            result = random.randint(0, self.num_classes - 1)
            if result != exclude:
                break

        return self.class_idx_to_class[result]

    def get_support_set(self, k):
        support_set = []
        support_set_classes = []
        for class_name, support_range in self.class_to_range.items():
            support_indexs = []
            for i in range(k):
                # Get Index of Support Image
                while True:
                    support_idx = random.randint(support_range[0], support_range[1])
                    if support_idx not in support_indexs:
                        break

                # Load Image
                img = self.__get_image__(support_idx)

                support_set.append(img)
                support_set_classes.append(class_name)
                support_indexs.append(support_idx)
        return mx.array(support_set), support_set_classes

    def __len__(self):
        return self.num_images


class TrainDataLoader:
    # Used for Training and Validation of Binary Classification
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(dataset))

    def __iter__(self):
        self.current = 0
        np.random.shuffle(self.indexes)  # Shuffle data each epoch
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        indexes = self.indexes[self.current:self.current+self.batch_size]
        anchors = []
        contrasts = []
        labels = []
        for idx in indexes:
            anchor, contrast, label = self.dataset[idx]
            anchors.append(anchor)
            contrasts.append(contrast)
            labels.append(label)
        self.current += self.batch_size
        return mx.array(anchors), mx.array(contrasts), mx.array(labels)

    def get_support(self, k):
        return self.dataset.get_support_set(k)


class EvalDataLoader:
    # Used for testing K-Shot Learning
    def __init__(self, dataset, k=1, batch_size=32):
        self.dataset = dataset
        self.indexes = np.arange(len(dataset))
        self.k = k
        self.batch_size = batch_size
        self.support_set, self.support_set_classes = self.dataset.get_support_set(self.k)

    def __iter__(self):
        self.current = 0
        self.support_set, self.support_set_classes = self.dataset.get_support_set(self.k)
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration

        indexes = self.indexes[self.current:self.current + self.batch_size]
        anchors = []
        anchor_classes = []
        for idx in indexes:
            anchor, anchor_class = self.dataset[idx]
            anchors.append(anchor)
            anchor_classes.append(anchor_class)
        self.current += self.batch_size
        return mx.array(anchors), np.array(anchor_classes)

    def get_support(self, k):
        return self.dataset.get_support_set(k)

# from glld import Display
# from glld import LossLoader
import matplotlib.pyplot as plt

import numpy as np


class ClopemaLoader:
    paths_x = {}
    paths_y = {}

    categories = []

    landmark_colors = []
    landmark_names = []

    # bb_size = 16  # 29

    def __init__(self):
        # LossLoader.__init__(self)
        self.paths_x = {
            'train': '/home/danfergo/SIG/Code/Experiments/data/clopema/train/data_x.npy',
            'val': '/home/danfergo/SIG/Code/Experiments/data/clopema/validation/data_x.npy'
        }

        self.paths_y = {
            'train': '/home/danfergo/SIG/Code/Experiments/data/clopema/train/data_y.npy',
            'val': '/home/danfergo/SIG/Code/Experiments/data/clopema/validation/data_y.npy'
        }
        self.init_landmarks()
        self.categories = ['pants', 'polo', 'hoody', 'tshirt', 'tshirt-long', 'polo-long', 'towel', 'skirt', 'bluse']

    def get_paths(self, split):
        return self.paths_x[split], self.paths_y[split]

    def n_landmark_cats(self):
        return len(self.landmark_names)

    def n_garment_cats(self):
        return len(self.categories)

    def init_landmarks(self):
        self.landmark_names.append('null')
        self.landmark_colors.append((0, 0, 0))

        # pants
        self.landmark_names.append('left-leg-outer')
        self.landmark_colors.append((0, 0, 200))
        self.landmark_names.append('left-leg-inner')
        self.landmark_colors.append((0, 0, 255))
        self.landmark_names.append('crotch')
        self.landmark_colors.append((0, 255, 150))
        self.landmark_names.append('right-leg-inner')
        self.landmark_colors.append((0, 255, 0))
        self.landmark_names.append('rigth-leg-outer')
        self.landmark_colors.append((0, 200, 0))
        self.landmark_names.append('top-right')
        self.landmark_colors.append((0, 200, 200))
        self.landmark_names.append('top-left')
        self.landmark_colors.append((0, 255, 255))

        # tshirt
        # 'bottom-left':
        # 'bottom-right':
        # 'right-armpit':
        self.landmark_names.append('right-sleave-inner')
        self.landmark_colors.append((255, 255, 0))
        self.landmark_names.append('right-sleave-outer')
        self.landmark_colors.append((150, 150, 0))
        # 'right-shoulder':
        # 'neckline-right':
        # 'neckline-left':
        # 'left-shoulder':
        self.landmark_names.append('left-sleave-outer')
        self.landmark_colors.append((255, 255, 255))
        self.landmark_names.append('left-sleave-inner')
        self.landmark_colors.append((200, 200, 200))
        # 'left-armpit':

        # tshirt-long
        # 'bottom-left':
        # 'bottom-right':
        # 'right-armpit':
        # 'right-sleave-inner':
        # 'right-sleave-outer':
        # 'right-shoulder':
        # 'neckline-right':
        # 'neckline-left':
        # 'left-shoulder':
        # 'left-sleave-outer':
        # 'left-sleave-inner':
        # 'left-armpit':


        # towel
        # 'bottom-left')
        # self.landmark_colors.append((255, 255, 0))
        # 'bottom-right')
        # self.landmark_colors.append((200, 200, 0))
        # 'top-right')
        # self.landmark_colors.append((150, 150, 0))
        # 'top-left')
        # self.landmark_colors.append((100, 100, 0))

        # skirt
        # 'bottom-left':
        # 'bottom-right':
        # 'top-right':
        # 'top-left':

        # hoody
        # 'bottom-left':
        # 'bottom-right':
        # 'right-armpit':
        # 'right-sleave-inner':
        # 'right-sleave-outer':
        # 'right-shoulder':
        self.landmark_names.append('hood-right')
        self.landmark_colors.append((255, 0, 255))
        self.landmark_names.append('hood-top')
        self.landmark_colors.append((200, 0, 200))
        self.landmark_names.append('hood-left')
        self.landmark_colors.append((150, 0, 150))
        # 'left-shoulder':
        # 'left-sleave-outer':
        # 'left-sleave-inner':
        # 'left-armpit':

        # polo
        # 'bottom-left':
        # 'bottom-right':
        # 'right-armpit':
        # 'right-sleave-inner':
        # 'right-sleave-outer':
        # 'right-shoulder':
        # 'neckline-right':
        # 'collar-rigth')
        # self.landmark_colors.append((200, 0, 200))
        # 'collar-left')
        # self.landmark_colors.append((150, 0, 150))
        # 'neckline-left':
        # 'left-shoulder':
        # 'left-sleave-outer':
        # 'left-sleave-inner':
        # 'left-armpit':

        # polo-long
        # 'bottom-left':
        # 'bottom-right':
        # 'right-armpit':
        # 'right-sleave-inner':
        # 'right-sleave-outer':
        # 'right-shoulder':
        # 'neckline-right':
        # 'collar-rigth':
        # 'collar-left':
        # 'neckline-left':
        # 'left-shoulder':
        # 'left-sleave-outer':
        # 'left-sleave-inner':
        # 'left-armpit':

        # bluse
        # 'bottom-left':
        # 'bottom-middle':
        # 'bottom-right':
        # 'right-armpit':
        # 'right-shoulder':
        # 'neckline-right':
        # 'collar-right':
        # 'collar-left':
        # 'neckline-left':
        # 'left-shoulder':
        # 'left-armpit':

        # bluse
        self.landmark_names.append('bottom-left')
        self.landmark_colors.append((0, 0, 150))
        self.landmark_names.append('bottom-middle')
        self.landmark_colors.append((0, 0, 200))
        self.landmark_names.append('bottom-right')
        self.landmark_colors.append((0, 0, 255))
        self.landmark_names.append('right-armpit')
        self.landmark_colors.append((0, 255, 0))
        self.landmark_names.append('right-shoulder')
        self.landmark_colors.append((0, 200, 0))
        self.landmark_names.append('neckline-right')
        self.landmark_colors.append((0, 250, 0))
        self.landmark_names.append('collar-right')
        self.landmark_colors.append((0, 255, 250))
        self.landmark_names.append('collar-left')
        self.landmark_colors.append((0, 255, 255))
        self.landmark_names.append('neckline-left')
        self.landmark_colors.append((0, 150, 0))
        self.landmark_names.append('left-shoulder')
        self.landmark_colors.append((0, 200, 0))
        self.landmark_names.append('left-armpit')
        self.landmark_colors.append((0, 0, 255))

        self.landmark_names.append('fold_1')
        self.landmark_colors.append((0, 200, 255))
        self.landmark_names.append('fold_2')
        self.landmark_colors.append((0, 150, 200))

    #
    # def load_split(self, name):
    #     return LossLoader.load_split(self, name)

    def fetch_data(self, split_name):
        paths = self.get_paths(split_name)
        train_x = np.load(paths[0], encoding='bytes')
        train_y = np.load(paths[1], encoding='bytes')

        return train_x, train_y

    def fetch_multiple_data(self, split_names):
        datasets = []
        for split_name in split_names:
            datasets.append(self.fetch_data(split_name))
        return tuple(datasets)

    def image_shape(self):
        return (224, 224, 3)

    def balance_data(self, data, balance=None):
        if balance is None:
            return data

        balanced = []

        for i in range(len(balance)):
            balanced.append(repeat(filter_class(data, i), balance[i]))

        balanced_split_x = balanced[0][0]
        balanced_split_y = balanced[0][1]
        i = 0
        for b in balanced:
            if i == 0:
                i += 1
                continue

            balanced_split_x = np.concatenate((balanced_split_x, b[0]))
            balanced_split_y = np.concatenate((balanced_split_y, b[1]))
        return shuffle((balanced_split_x, balanced_split_y))

    def fetch_balanced_data(self, split_name, balance):
        return self.balance_data(self.fetch_data(split_name), balance)

    def fetch_multiple_balanced_data(self, split_names):
        splits = self.fetch_multiple_data(split_names)
        balanced_splits = []
        for split in splits:
            balanced_splits.append(self.balance_data(split))
        return balanced_splits


def gaussian_noise(data, strength=10):
    noise = np.random.normal(0, 1, data[0].shape)
    noised = np.add(data[0], strength * noise)
    min = np.min(noised)
    max = np.max(noised)

    # noise = np.ones(data[0].shape)
    # print(noise[0])
    # print(type(data[0][0][0][0][0]))
    noised = (((noised - min) / (max - min)) * 255).astype('uint8')
    # print(noised[0])
    return noised, data[1]


def hue_noise(data, strength=10):
    n = data[0].shape[0]
    noise = np.random.normal(0, 1, (data[0].shape[:1] + (3,)))
    noise = np.reshape(noise, (n,) + (1, 1) + (3,))
    noised = np.add(data[0], strength * noise)
    min = np.min(noised)
    max = np.max(noised)
    noised = (((noised - min) / (max - min)) * 255).astype('uint8')
    return noised, data[1]


def shuffle(data):
    indices = np.random.permutation(data[0].shape[0])
    return data[0][indices], data[1][indices]


def repeat(data, k=3):
    return np.repeat(data[0], k, axis=0), np.repeat(data[1], k, axis=0)
    # augmented_data = hue_noise(gaussian_noise(multiplied_data))
    # return multiplied_data[0][indices], multiplied_data[1][indices]


def augment(data):
    # print(data[0].shape)
    # multiplied_data = np.repeat(data[0], k, axis=0), np.repeat(data[1], k, axis=0)
    augmented_data = hue_noise(gaussian_noise(data))
    indices = np.random.permutation(data[0].shape[0])
    return augmented_data[0][indices], augmented_data[1][indices]


def mean_subtract(data):
    mean = np.mean(data[0], axis=(0, 1, 2), keepdims=True)
    return data[0] - mean, data[1]


def filter_class(data, class_idx):
    whr = np.where(data[1][:, 0] == class_idx)
    return data[0][whr], data[1][whr]


def filter_by_n_classes(data, class_idx):
    whr = np.where(data[1][:, 2].size == class_idx)

    return data[0][whr], data[1][whr]


def filter_by_n_landmarks(data, n_landmarks=None):
    if n_landmarks is None:
        n_landmarks = np.min([len(x) for x in data[1][:, 2]])

    idxs = [len(x) == n_landmarks for x in data[1][:, 2]]
    return data[0][idxs], data[1][idxs]


# def expand_landmarks(data):
#     def expand(x):
#         return np.reshape(np.array(x)[:, :2], np.array(x)[:, :2].size).tolist()
#
#     # print(data[1].shape)
#     # print(np.array(data[1][:, 2][0]).shape)
#     landmark_classes = [expand(x) for x in data[1][:, 2]]
#
#     data_1 = []
#     for x in range(len(landmark_classes)):
#         data_1.append([data[1][x][0], data[1][x][1], landmark_classes[])
#
#     # landmark_classes.reshape()
#     # print(landmark_classes.shape)
#     # data[1][:, :2]
#     # print(d, d.size).shape)
#     # for x in landmark_classes:
#     #     print(x.shape)
#     # print(landmark_classes.shape)
#     print(data[1][:, 0].shape)
#     print(data[1][:, 1].shape)
#     print(landmark_classes.shape)
#     # landmark_classes = np.expand_dims(landmark_classes, 1)
#     # print(landmark_classes.shape)
#     return data[0], np.array([data[ 1][:, 0], data[1][:, 1], landmark_classes])



def print_dataset_histo(dataset, label, data):
    print(label + ': ')
    for x in range(dataset.n_garment_cats()):
        print(dataset.categories[x] + ': \t\t\t\t' + str(
            len((filter_class(data, x))[0])))


def main():
    """ Used for 'unit testing'
    """
    dataset = ClopemaLoader()
    data = dataset.fetch_data('train')
    print(len(data[0]))
    # # data = augment(filter_by_n_landmarks(filter_class(dataset.fetch_data('val'), 0)))
    #
    # train = dataset.fetch_data('train')
    # val = dataset.fetch_data('val')
    #
    # print_dataset_histo(dataset, 'TRAIN', train)
    # print_dataset_histo(dataset, 'VALIDATION', val)
    #
    # print('\n\n\n\n')
    # print('AFTER BALANCE')
    #
    # data = dataset.fetch_multiple_balanced_data(['train', 'val'])
    #
    # print_dataset_histo(dataset, 'TRAIN', data[0])
    # print_dataset_histo(dataset, 'VALIDATION', data[1])







    data = hue_noise(gaussian_noise(data))
    # print(data[1].shape)
    # print(dataset[1].shape)
    for x in data[0]:
        plot = plt.imshow(x)
        plt.show()
        # plot = Display(dataset)
        #
        # x, rpn_cls, rpn_reg, roi_cls, roi_reg, cls, reg = dataset.load_split('val')
        #
        # plot.show_results((rpn_cls, rpn_reg, roi_cls, roi_reg, cls, reg), split_name='val')


if __name__ == '__main__':
    main()

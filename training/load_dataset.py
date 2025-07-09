import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as tdata
import PIL
from torchvision import tv_tensors
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import torch.optim as optim

random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Parse_Data(tdata.Dataset):
    def __init__(self, training_data, training_labels, transform=None, target_transform=None):
        self.img_labels = training_labels
        self.imgs = training_data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label

# Function for visualizing images
def show_image(img, title=None):
    plt.imshow(img.detach().cpu().permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_data():
    path = "/content/data/PetImages"
    raw = []
    splits = [0.7, 0.15, 0.15]  # train, validation, test
    fraction = 0.2    # Number in [0,1] that modulates how much of the dataset to process and use

    imgs_dog = [os.path.join(path, 'Dog', f) for f in os.listdir(os.path.join(path, 'Dog')) if os.path.isfile(os.path.join(path, 'Dog', f))]
    imgs_cat = [os.path.join(path, 'Cat', f) for f in os.listdir(os.path.join(path, 'Cat')) if os.path.isfile(os.path.join(path, 'Cat', f))]

    print(f"Number of Dogs: {len(imgs_dog)} - Number of Cats: {len(imgs_cat)}") # We can see that the classes are balanced

    # Load the images into a list
    for i in range(int(len(imgs_dog)*fraction)):
        try:
            raw.append([tv_tensors.Image(PIL.Image.open(imgs_dog[i]).convert('RGB')), 0])
        except:
            continue
    for i in range(int(len(imgs_cat)*fraction)):
        try:
            raw.append([tv_tensors.Image(PIL.Image.open(imgs_cat[i]).convert('RGB')), 1])
        except:
            continue

    random.shuffle(raw)

    train_raw = raw[:int(splits[0] * len(raw))]
    valid_raw = raw[int(splits[0] * len(raw)):(int(splits[0] * len(raw)) + int(splits[1] * len(raw)))]
    test_raw = raw[(int(splits[0] * len(raw)) + int(splits[1] * len(raw))):]

    # Verify that the classes remain balanced after shuffling and splitting
    num_class = [0, 0]
    for _, label in train_raw:
        num_class[label] += 1
    print(f"Number of Dogs: {num_class[0]} - Number of Cats: {num_class[1]}") # We can see that the classes are still balanced
    return train_raw, valid_raw, test_raw


def preprocess(train, valid, test, H_resize=224, W_resize=224, batch_size=16):
    transform_train = v2.Compose([v2.RandomRotation(30),
                                  v2.RandomResizedCrop((H_resize, W_resize)),
                                  v2.RandomHorizontalFlip(),
                                  v2.ConvertImageDtype(torch.float32),
                                  v2.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

    transform_base = v2.Compose([v2.Resize((H_resize, W_resize)),
                                 v2.CenterCrop((H_resize, W_resize)),
                                 v2.ConvertImageDtype(torch.float32),
                                 v2.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

    tr_data, v_data, t_data = [], [], []
    tr_label, v_label, t_label = [], [], []
    for sample in train:
        tr_data.append(transform_train(sample[0]))
        tr_label.append(sample[1])
    for sample in valid:
        v_data.append(transform_base(sample[0]))
        v_label.append(sample[1])
    for sample in test:
        t_data.append(transform_base(sample[0]))
        t_label.append(sample[1])

    train_args = {'batch_size': batch_size}
    val_args = {'batch_size': batch_size}
    test_args = {'batch_size': batch_size}
    train_loader = torch.utils.data.DataLoader(Parse_Data(tr_data, tr_label), **train_args)
    valid_loader = torch.utils.data.DataLoader(Parse_Data(v_data, v_label), **val_args)
    test_loader = torch.utils.data.DataLoader(Parse_Data(t_data, t_label), **test_args)
    return train_loader, valid_loader, test_loader
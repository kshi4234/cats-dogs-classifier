import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Classifier
from cd_model import Resnet50Model
from load_dataset import load_data, preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set convolutional parameters
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
import sys

# Set training parameters
epochs = 20
lr = 5e-6
momentum = 0.001
reg = 0.1
batch_size = 4

# Save model path
PATH = '/content/models/best_model.pt'

def train(model, train_data, valid_data, loss_fn, optimizer, scheduler):
  prev_val_acc = 0
  for epoch in tqdm(range(epochs)):
    model.train()
    for batch_idx, (batch, target) in enumerate(train_data):
        # target = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float32)
        batch, target = batch.to(device), target.to(device)

        optimizer.zero_grad()

        logits, probs = model(batch)

        cost = loss_fn(logits, target)

        cost.backward()
        optimizer.step()

        # print(f"Predictions: {pred} \t-\t Target Labels: {target}")
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                  %(epoch+1, epochs, batch_idx,
                    len(train_data), cost))
    # scheduler.step()  # Use the scheduler each epoch

    model.eval()
    with torch.set_grad_enabled(False):
      val_acc = validate(model, valid_data, epoch, loss_fn)
      train_acc = validate(model, train_data, epoch, loss_fn)
      print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%%' % (
          epoch+1, epochs,
          train_acc, val_acc))
      if val_acc > prev_val_acc:
        prev_val_acc = val_acc
        torch.save(model.state_dict(), PATH)

def validate(model, data, epoch, loss_fn):
  correct_pred, num_examples = 0, 0
  for i, (features, targets) in enumerate(data):
    features = features.to(device)
    targets = targets.to(device)

    logits, probas = model(features)
    # print('logits:', logits.shape)
    # print('probs:', probas.shape)
    _, predicted_labels = torch.max(probas, 1)
    if not i % 50:
      print('preds:', predicted_labels)
    # print('truth:', targets)
    # sys.exit()
    num_examples += targets.size(0)
    correct_pred += (predicted_labels == targets).sum()
  return correct_pred.float()/num_examples * 100


# Set convolutional parameters
H, W = 256, 256
num_channels = [64, 128, 256, 512]

# Although my implementation of ResNet50 is exact, without pretrained weights
# it performs poorly. Using the parameters from ResNet50 trained on ImageNet,
# performance is much more robust and has stronger generalization.
def main():
    # model = Classifier(H=H, W=W,
    #                   num_blocks=[3, 4, 6, 3], num_channels=num_channels,
    #                   num_classes=2, grayscale=False)h
    model = Resnet50Model()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # dense_loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)

    train_raw, valid_raw, test_raw = load_data()
    print('DATA LOADED!!!')
    train_data, valid_data, test_data = preprocess(train_raw, valid_raw, test_raw, H_resize=H, W_resize=W, batch_size=batch_size)
    print('DATA PROCESSED!!!')

    train(model, train_data=train_data, valid_data=valid_data, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)

if __name__ == '__main__':
    main()
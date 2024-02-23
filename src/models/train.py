from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn 
import os
import yaml
import wandb
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, Accuracy, BinaryF1Score, F1Score, BinaryAUROC, BinaryConfusionMatrix, MulticlassConfusionMatrix, BinaryPrecision, BinaryRecall
import numpy as np
from datetime import datetime
import time
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.ImageClassificationDataset import ImageClassificationDataset
from src.models.model import SimpleCNN, XrayCNN, XrayCNN_mini
from src.data.preprocessing import EqualizeClahe

def make(cfg):
    # 1 - prepare data
    train_root = os.path.join(cfg.data_root, "Train")

    if not cfg.data_augm:
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.GaussianBlur(5), #TODO add config to choose if to preprocess or not
            # EqualizeClahe(), #TODO add config to choose if to preprocess or not
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
    else:
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomApply([transforms.CenterCrop(128)], p = 0.5),
            transforms.RandomPerspective(distortion_scale = 0.2, p = 0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )

    full_train_dataset = ImageClassificationDataset(train_root, transform=transform)

    # split train into train and val
    train_ind, val_ind = train_test_split(
        range(len(full_train_dataset)), 
        stratify=full_train_dataset.classes,
        test_size=0.2,
        random_state=1)

    train_dataset = Subset(full_train_dataset, train_ind)
    val_dataset = Subset(full_train_dataset, val_ind)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle = True)

    # 2 - prepare model
    if "model1" in cfg.name:
        model = SimpleCNN()
    elif "model2" in cfg.name:
        model = XrayCNN()
    elif "model3" in cfg.name:
        model = XrayCNN_mini()
    else:
        NotImplementedError()


    # 3 - prepare loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = cfg.lr, momentum=0.9)

    return model, train_loader, val_loader, criterion, optimizer

def validate(model, val_loader, criterion):
    # print("Starting validation...")
    with torch.no_grad():
        # for data in tqdm(val_loader):
        for data in val_loader:
            images, labels = data

            # only forward step
            outputs = model(images)
            loss = criterion(outputs, labels)

    return loss

def train(model, train_loader, val_loader, criterion, optimizer, cfg):
    print("Starting training...")
    print("cfg=", cfg, "type=", type(cfg))
    wandb.watch(model, criterion, log="all", log_freq=10)

    print("cfg.epochs=", cfg.epochs)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)

    for epoch in tqdm(range(cfg.epochs)):
        # print(f"Epoch {epoch} / {cfg.epochs}")
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            # images, labels = images.to(device), labels.to(device)

            # zero the gradient
            optimizer.zero_grad()

            # forward and backward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log
            wandb.log({"epoch": epoch, "train_loss": loss.item()}) #, "train_step": i
        
        # validate
        # model = model.cpu()
        model.eval()
        val_loss = validate(model, val_loader, criterion)
        
        # log
        wandb.log({"epoch": epoch, "val_loss": val_loss.item()})

    print("Training finished!")

def evaluate(model, test_loader):
    print("Starting testing ...")
    model.eval()

    acc = BinaryAccuracy()
    f1score = BinaryF1Score()
    auroc = BinaryAUROC()
    conf_matrix = BinaryConfusionMatrix()
    prec = BinaryPrecision()
    recall = BinaryRecall()

    with torch.no_grad():

        for data in tqdm(test_loader):
            images, labels = data

            # only forward step
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # calculate metrics
            acc.update(preds, labels)
            f1score.update(preds, labels)
            auroc.update(preds, labels)
            conf_matrix.update(preds, labels)
            prec.update(preds, labels)
            recall.update(preds, labels)

    total_acc = acc.compute().item()
    total_f1score = f1score.compute().item()
    total_auroc = auroc.compute().item()
    total_conf_matrix = conf_matrix.compute()
    total_prec = prec.compute().item()
    total_recall = recall.compute().item()

    stats = {
        "Accuracy": total_acc, 
        "Precision": total_prec,
        "Recall": total_recall, 
        "F1Score": total_f1score, 
        "AUROC":total_auroc, 
        "Confusion_Matrix_array": np.array(total_conf_matrix)
    }

    wandb.log(stats)

    # log confusion matrix as an image
    conf_matrix_np = total_conf_matrix.cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_np, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Binary Confusion Matrix")
    plt.tight_layout()
    wandb.log({"Binary Confusion Matrix": wandb.Image(plt)})

    return stats


def train_pipeline(hyperparameters):

    with wandb.init(project="xray-classifier", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, val_loader, criterion, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, train_loader, val_loader, criterion, optimizer, config)

        # evaluate on validation set
        stats = evaluate(model, val_loader)
 
        # Save the model
        print("Saving a model!")
        # date = datetime.now().strftime("%d-%m-%Y")
        model_path = f"models/{config.name}_{wandb.run.name}_bs{config.batch_size}_e{config.epochs}_{time.time()}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.log({"Model_path": model_path})
    
    wandb.finish()

    return model, stats


if __name__ == "__main__":
    # To run this file from the root folder (xray_classifier): python -m src.models.model1.train

    # TODO read config as command line argument (wrt cwd - xray_Classifier)
    config_name = sys.argv[1]
    config_path = f"configs/{config_name}.yaml"

    with open(config_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    print(hyperparameters)

    wandb.login()
    model, stats = train_pipeline(hyperparameters)

    print(stats)
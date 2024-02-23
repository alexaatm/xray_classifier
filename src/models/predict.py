import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall
from tqdm import tqdm
import wandb
import numpy as np
import sys
from src.data.ImageClassificationDataset import ImageClassificationDataset
from src.data.ImageDataset import ImageDataset
from src.models.model import SimpleCNN
from torchvision.transforms import transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def make(cfg, model_path):
    # 1 - prepare data
    test_root = os.path.join(cfg.data_root)

    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )

    if not cfg.eval:
        test_dataset = ImageDataset(test_root, transform=transform)
    else:
        test_dataset = ImageClassificationDataset(test_root, transform=transform)

    # loaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = cfg.batch_size, shuffle = False, drop_last = False)

    # 2 - prepare model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    return model, test_loader

def predict(model, test_loader):
    print("Starting testing ...")
    model.eval()
    
    result = {}

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, names = data

            # only forward step
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for n, p in zip(names, preds.tolist()):
                result[n] = p

    return result


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

def test_pipeline(hyperparameters, model_path):

    with wandb.init(project="xray-classifier", config=hyperparameters, tags = ["test"]):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        model, test_loader = make(config, model_path)

        # and use them to predict and/or evaludate
        if config.eval:
            # evaluate on test set and log metrics
            print(model)
            stats = evaluate(model, test_loader)
            print(f'Metrics: {stats}')
        else:
            # just predict labels
            preds = predict(model, test_loader)
            print(f'Predictions: {preds}')                    
    
    wandb.finish()

if __name__ == "__main__":
    config_name = sys.argv[1]
    model_name = sys.argv[2]
    config_path = f"configs/{config_name}.yaml"
    model_path = f"models/{model_name}"


    with open(config_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    print(hyperparameters)

    wandb.login()
    test_pipeline(hyperparameters, model_path)
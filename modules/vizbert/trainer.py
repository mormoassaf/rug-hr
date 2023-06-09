#########################################################
#                                                       #
#   This file contains the SEGNET class, which is used  #
#   to train and test the segmentation network.         #
#     Created by Mo Assaf moassaf42@gmail.com (2020)    #
#########################################################

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import monai
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from tqdm import tqdm
from .losses import ContrastiveLoss


class VizBERTTrainer():

    def __init__(self,
                 model_name,
                 device,
                 metric=None,
                 max_sequence_length=256,
                 model=None) -> None:
        # Properties 
        self.model_name = model_name
        self.device = device
        self.metric = metric
        self.post_pred = None
        self.post_label = None
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.timesteps = torch.arange(0, max_sequence_length, dtype=torch.long).to(device)

    def __call__(self, x, batch_size=16):
        return self.predict(x, batch_size=batch_size)

    def get_model(self):
        return self.model

    def load_model(self, name=None):
        if not name:
            name = self.model_name
        state = torch.load(os.path.join(os.getcwd(), name))
        self.model.load_state_dict(state)
        self.model.to(self.device)
        return self.model

    def save_model(self, name=None, out_dir="models", alias=""):
        if not name:
            name = self.model_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(self.model.state_dict(), os.path.join(os.getcwd(), out_dir, alias + name))

    def get_model_summary(self):
        return self.model.eval()

    def loss_func(self):
        # return torch.nn.CrossEntropyLoss()
        return ContrastiveLoss()

    def metric_func(self):
        if self.metric:
            evaluate = self.metric
        else:
            evaluate = monai.metrics.DiceMetric(include_background=True, reduction="mean")
        return evaluate

    def train(
            self,
            trainloader: torch.utils.data.DataLoader,
            valloader: torch.utils.data.DataLoader,
            log,
            epochs: int,
            learning_rate=1e-4,
            checkpoints=True,
            schedule=False,
            optimizer_name="adam",
    ) -> None:
        print(
            f"epochs={epochs}; N={len(trainloader.dataset)}; batches={len(trainloader)}; learning_rate={learning_rate}")

        # Get model and loss function
        net = self.model
        criterion = self.loss_func()

        # Create optimizer
        optimizer = None
        if (optimizer_name == "adam"):
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        elif (optimizer_name == "rms"):
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8,
                                            momentum=0.9)
        elif (optimizer_name == "adamw"):
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            raise ValueError("Unknown optimizer")

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2,
                                                                  min_lr=0.00001)

        train_losses = []
        val_losses = []
        metrics = []

        _, best_metric_score = self.validate(valloader)
        best_metric_score = 0
        print(f"current best metric: {best_metric_score}")

        for epoch in range(1, epochs + 1):
            net.train()
            avg_loss = 0.0
            n = 0

            # Load batches and apply passes
            epoch_iterator = tqdm(trainloader, dynamic_ncols=True)
            for step, batch in enumerate(epoch_iterator):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)

                probabilities = net(images)
                loss = criterion(probabilities, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item()
                n += 1

            avg_loss /= n

            # Evaluate
            test_loss, metric = self.validate(valloader)
            train_losses.append(avg_loss)
            val_losses.append(test_loss)
            metrics.append(metric)

            # Update scheduler 
            if (schedule):
                lr_scheduler.step(test_loss)
            lr = optimizer.param_groups[0]['lr']

            log({
                "epoch": epoch,
                "loss_train": avg_loss,
                "loss_test": test_loss,
                "learning_rate": lr,
                "metric": metric
            })

            print(
                "epoch {}/{}; train_loss={}; test_loss={}; lr={}; metric={}".format(epoch, epochs,
                                                                                    avg_loss,
                                                                                    test_loss, lr,
                                                                                    metric))

            # Save model if it has improved
            if checkpoints and metric >= best_metric_score:
                best_metric_score = metric
                self.save_model()

        return train_losses, val_losses, metrics

    def validate(
            self,
            valloader: torch.utils.data.DataLoader,
    ):
        net = self.model
        net.eval()
        criterion = self.loss_func()
        loss = 0.0
        n = 0
        metric = 0.0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(valloader), total=len(valloader)):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                probabilities = net(images)
                loss += criterion(probabilities, labels).item()
                metric += torch.nn.functional.cosine_similarity(probabilities, labels,
                                                                dim=-1).mean()
                n += batch["image"].shape[0]

        return loss / n, metric / n

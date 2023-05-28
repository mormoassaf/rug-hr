
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


class SEGNET():
    
    def __init__(self, 
        model_name, 
        device, 
        load=True, 
        metric=None, 
        model=None,
        n_classes=28,
        img_size=None) -> None:
        # Properties 
        self.model_name = model_name
        self.device = device
        self.metric = metric
        self.img_size = img_size
        self.n_classes = n_classes
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=n_classes)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_classes)])
        self.model = model
        if load:
            self.model = self.load_model()
        else:
            if not model:
                raise ValueError("net is required when not loading!")

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
        return monai.losses.DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

    def metric_func(self):
        if self.metric:
            evaluate = self.metric
        else:
            evaluate = monai.metrics.DiceMetric(include_background=True, reduction="mean")
        return evaluate

    def predict(self, scan: np.array, batch_size=16):
        self.model.eval()
        with torch.no_grad():
            x = torch.Tensor(scan).to(device=self.device, dtype=torch.float32)
            x = torch.unsqueeze(x, 1).cuda()
            x = self.infere(x)
            x = self.post_trans(x)
            x = torch.squeeze(x, 1)
            return x.cpu()

    def infere(self, x: torch.Tensor, batch_size=4):
        return sliding_window_inference(x, self.img_size, batch_size, self.model, overlap=0.6)
    
    def one_epoch(self, 
                  net: torch.nn.Module, 
                  optimizer, 
                  criterion, 
                  loader: torch.utils.data.DataLoader, 
                  scaler=None,
                  clip_gradients=None
                  ):
        net.train()
        avg_loss = 0.0
        n = 0

        # Load batches and apply passes
        epoch_iterator = tqdm(loader, dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            
            images, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
            logits = net(images)
            assert labels.max() < self.n_classes, f"SEGNET :: ERROR :: label max is {labels.max()} and not in range [0, {self.n_classes}-1]"

            if scaler == None:
                loss = criterion(logits, labels)
                loss.backward()
                if clip_gradients:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradients)
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if clip_gradients:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            avg_loss += loss.item()

            n+=1

        avg_loss /= n
        
        return avg_loss
        
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
        use_amp=False,
        clip_gradients=None
    ) -> None:
        print(f"epochs={epochs}; N={len(trainloader.dataset)}; batches={len(trainloader)}; learning_rate={learning_rate}")

        # Get model and loss function
        net = self.model
        criterion = self.loss_func()
        grad_scaler = torch.cuda.amp.GradScaler()

        # Create optimizer
        optimizer = None
        if (optimizer_name == "adam"):
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        elif (optimizer_name == "rms"):
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        elif (optimizer_name == "adamw"):
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            raise ValueError("Unknown optimizer")

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, min_lr=0.00001)
        
        train_losses = []
        val_losses = []
        metrics = []

        best_metric_score = 0

        for epoch in range(1, epochs+1):

            # Evaluate
            test_loss, metric = self.validate(valloader)
        
            # Save model if it has improved
            if checkpoints and metric >= best_metric_score and epoch > 1:
                best_metric_score = metric
                self.save_model()

            avg_loss = self.one_epoch(net, optimizer, criterion, trainloader, scaler=grad_scaler if use_amp else None, clip_gradients=clip_gradients)
            
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

            train_losses.append(avg_loss)
            val_losses.append(test_loss)
            metrics.append(metric)

            print("epoch {}/{}; train_loss={}; test_loss={}; lr={}; metric={}".format(epoch, epochs, avg_loss, test_loss, lr, metric))
            
        return train_losses, val_losses, metrics

    def validate(
        self,
        valloader: torch.utils.data.DataLoader,
        decollate_batch=decollate_batch,
        free_memory=False
    ):
        net = self.model
        net.eval()
        criterion = self.loss_func()
        metric_func = self.metric_func()
        loss = 0.0
        n = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(valloader), total=len(valloader)):

                images, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
                logits = self.infere(images)
                loss += criterion(logits, labels).item()
                
                if decollate_batch is not None:
                    val_labels_list = decollate_batch(labels)
                    val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                    val_outputs_list = decollate_batch(logits)
                    val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                    y_pred = val_output_convert
                    y = val_labels_convert
                else:
                    y_pred = self.post_pred(logits)
                    y = self.post_label(labels)
                # free samples to reduce memory consumption
                if free_memory:
                    del images, labels, logits
                    torch.cuda.empty_cache()
                metric_func(y_pred=y_pred, y=y)
                n+=1
        metric = metric_func.aggregate().item()
        metric_func.reset()
        return loss / n, metric

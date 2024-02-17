import torch
import torch.nn.functional as F
import lightning as L


class ModelTrainer(L.LightningModule):
    def __init__(self, classifier, learning_rate, debug=False, use_discriminative_lr: bool = True, lr_mult: int= 0.98):
        super().__init__()
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.val_losses = []
        self.top_loss_images = []
        self.top_loss_true_labels = []
        self.top_loss_pred_labels = []
        self.debug = debug
        self.use_discriminative_lr= use_discriminative_lr
        self.lr_mult    = lr_mult

    def forward(self, img):
        output = self.classifier(img)
        return output

    def parameters(self, recurse: bool = True):
        if self.use_discriminative_lr:
            parameters =  self._setup_discriminative_lr()
        else:
            parameters = super().parameters(recurse=recurse)

        return parameters

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    def _setup_discriminative_lr(self):
        # save layer names
        layer_names = []
        for idx, (name, param) in enumerate(self.classifier.named_parameters()):
            layer_names.append(name)

        layer_names.reverse()
        lr = self.learning_rate
        lr_mult = self.lr_mult

        parameters = []

        # store params & learning rates
        for idx, name in enumerate(layer_names):
            # display info
            if self.debug:
                print(f'{idx}: lr = {lr:.6f}, {name}')

            # append layer parameters
            parameters += [{'params': [p for n, p in self.classifier.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]

            # update learning rate
            lr *= lr_mult

        return parameters

    def _make_prediction(self, batch, batch_idx):
        img_batch, label_batch = batch
        y_hat = self(img_batch)
        loss = F.cross_entropy(y_hat, label_batch)
        probabilities = F.softmax(y_hat, dim=1)
        pred_labels = torch.argmax(probabilities, dim=1)

        accuracy = torch.sum(label_batch == pred_labels).item() / label_batch.size(0)
        return {'loss': loss, "accuracy": accuracy, 'predicted_labels': pred_labels, 'true_labels': label_batch}

    def training_step(self, train_batch, batch_idx):
        res = self._make_prediction(train_batch, batch_idx)

        self.log("loss", res["loss"], on_epoch=True, on_step=False, prog_bar=True)
        self.log("accuracy", res["accuracy"], on_epoch=True, on_step=False, prog_bar=True)
        return res

    def validation_step(self, valid_batch, batch_idx):

        res = self._make_prediction(valid_batch, batch_idx)
        self.log("val_loss", res["loss"], on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_accuracy", res["accuracy"], on_epoch=True, on_step=False, prog_bar=True)

        # Check if the prediction is incorrect
        incorrect_indices = (res["predicted_labels"] != res["true_labels"]).nonzero(as_tuple=True)[0]

        # Store only the top 5 incorrect predictions
        for idx in incorrect_indices:
            if len(self.top_loss_images) < 5:
                top_loss_img = valid_batch[0][idx]
                top_loss_pred_label = res["predicted_labels"][idx]
                top_loss_true_label = res["true_labels"][idx]
                self.top_loss_true_labels.append(top_loss_true_label.item())
                self.top_loss_pred_labels.append(top_loss_pred_label.item())
                self.top_loss_images.append(top_loss_img)

        return res
import random

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import lightning as L


class TrainingModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.inner_model = model
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, questions, labels = batch
        output = self.inner_model(images, questions)
        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        values = {"train_loss": loss, "train_acc": acc}  # add more items if needed
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['val', 'systematic_val'][dataloader_idx]
        images, questions, labels = batch
        output = self.inner_model(images, questions)
        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        values = {f"{dl_name}_loss": loss, f"{dl_name}_acc": acc}  # add more items if needed
        self.log_dict(values)

    def test_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['test', 'systematic_test'][dataloader_idx]
        images, questions, labels = batch
        output = self.inner_model(images, questions)
        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        values = {f"{dl_name}_loss": loss, f"{dl_name}_acc": acc}  # add more items if needed
        self.log_dict(values)

    def configure_optimizers(self):
        if self.config.optimizer == 'sgd':
            optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.config.lr)
        else:
            optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.config.lr)

        return optimizer


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        patch_size = config.patch_height * config.patch_width * 3
        self.word_embedding = nn.Embedding(config.n_tokens, config.d_hidden)
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_height, p2=config.patch_width),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, config.d_hidden),
            nn.LayerNorm(config.d_hidden),
        )

        self.pos_embedding = nn.Parameter(torch.randn(config.n_patches + config.max_question_size, 1, config.d_hidden))
        self.type_image_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_question_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))

        self.classifier = nn.Linear(config.d_hidden, config.n_outputs)

    def forward(self, image, question):
        embedded_image = self.patch_embedding(image).transpose(1,0)
        embedded_question = self.word_embedding(question).transpose(1,0)

        embedded_image = embedded_image + self.type_image_embedding
        embedded_question = embedded_question + self.type_question_embedding

        i_pad_mask = torch.zeros_like(embedded_image[:,:,0], dtype=torch.bool).transpose(1,0)
        q_pad_mask = question == self.config.pad_idx
        pad_mask = torch.cat((i_pad_mask, q_pad_mask), dim=1)

        combined_embedding = torch.cat((embedded_question, embedded_image))
        transformer_input = combined_embedding + self.pos_embedding

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        return self.classifier(transformer_output[0,:,:])


class TextualModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.aug_zero = config.aug_zero
        n_tokens = config.n_tokens * (1 + self.aug_zero)
        self.word_embedding = nn.Embedding(n_tokens, config.d_hidden)

        self.pos_embedding = nn.Parameter(torch.randn(config.max_scene_size + config.max_question_size, 1, config.d_hidden))
        self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_question_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))

        self.classifier = nn.Linear(config.d_hidden, config.n_outputs)

    def forward(self, scene, question):
        if self.training and self.aug_zero:
            aug_zero_offset = random.randint(0, self.aug_zero)
        else:
            aug_zero_offset = 0

        embedded_scene = self.word_embedding(scene + aug_zero_offset).transpose(1,0)
        embedded_question = self.word_embedding(question + aug_zero_offset).transpose(1,0)

        embedded_scene = embedded_scene + self.type_scene_embedding
        embedded_question = embedded_question + self.type_question_embedding

        s_pad_mask = scene == self.config.pad_idx
        q_pad_mask = question == self.config.pad_idx
        pad_mask = torch.cat((s_pad_mask, q_pad_mask), dim=1)

        combined_embedding = torch.cat((embedded_question, embedded_scene))
        transformer_input = combined_embedding + self.pos_embedding

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        return self.classifier(transformer_output[0,:,:])


class TextualBiEncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        patch_size = config.patch_height * config.patch_width * 3
        self.word_embedding = nn.Embedding(config.n_tokens, config.d_hidden)
        # self.patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_height, p2=config.patch_width),
        #     nn.LayerNorm(patch_size),
        #     nn.Linear(patch_size, config.d_hidden),
        #     nn.LayerNorm(config.d_hidden),
        # )

        self.pos_embedding = nn.Parameter(max(torch.randn(config.max_scene_size, config.max_question_size), 1, config.d_hidden))
        # self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        # self.type_question_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))

        self.classifier = nn.Linear(config.d_hidden, config.n_outputs)

    def forward(self, image, question):
        embedded_image = self.patch_embedding(image).transpose(1,0)
        embedded_question = self.word_embedding(question).transpose(1,0)

        embedded_image = embedded_image + self.type_image_embedding
        embedded_question = embedded_question + self.type_question_embedding

        i_pad_mask = torch.zeros_like(embedded_image[:,:,0], dtype=torch.bool).transpose(1,0)
        q_pad_mask = question == self.config.pad_idx
        pad_mask = torch.cat((i_pad_mask, q_pad_mask), dim=1)

        combined_embedding = torch.cat((embedded_question, embedded_image))
        transformer_input = combined_embedding + self.pos_embedding

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        return self.classifier(transformer_output[0,:,:])

import random

import torch
# torch.fx.wrap('len')
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import lightning as L

from transformers import ViTImageProcessor, ViTModel


def build_model(config):
    if config.multimodal_pretraining or config.multimodal_training:
        print("Creating MultimodalModel model")
        model = MultimodalModel(config)
    elif config.image_pretraining:
        print("Creating MultimodalModelForImagePretraining model")
        model = MultimodalModelForImagePretraining(config)
    elif config.use_txt_scene:
        print("Creating TextualModel model")
        model = TextualModel(config)
    else:
        print("Creating MultimodalModel model")
        model = MultimodalModel(config)
        # model = Model(config)

    if config.multimodal_pretraining:
        print("Creating MultimodalPretrainingModel training model")
        training_model = MultimodalPretrainingModel(model, config)
    elif config.image_pretraining:
        print("Creating ImageMaskingPretrainingModel training model")
        training_model = ImageMaskingPretrainingModel(model, config)
    else:
        # print("Creating TrainingModel training model")
        # training_model = TrainingModel(model, config)
        print("Creating MultimodalPretrainingModel training model")
        training_model = MultimodalPretrainingModel(model, config)

    if config.start_from:
        resume_from_path = f"outputs/{config.start_from}/last.ckpt"
        training_model.load_state_dict(torch.load(resume_from_path)['state_dict'])

    return training_model


class TrainingModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.inner_model = model
        self.save_hyperparameters()
        self.exp_prefix = ''

    def set_exp_prefix(self, prefix):
        self.exp_prefix = prefix

    def pack_logging_values(self, metrics, prefix):
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def training_step(self, batch, batch_idx):
        if self.config.multimodal_training:
            images, scenes, questions, labels = batch
            output = self.inner_model(images, scenes, questions)
        else:
            images, questions, labels = batch
            output = self.inner_model(images, questions)

        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        values = {"loss": loss, "acc": acc}  # add more items if needed
        values = self.pack_logging_values(values, 'train')
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['val', 'systematic_val',
                   'color_val', 'color_systematic_val', 
                   'shape_val', 'shape_systematic_val', 
                   'color_common_systematic_val', 'shape_common_systematic_val'][dataloader_idx]

        if self.config.multimodal_training:
            images, scenes, questions, labels = batch
            output = self.inner_model(images, scenes, questions)
        else:
            images, questions, labels = batch
            output = self.inner_model(images, questions)

        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {"loss": loss, "acc": acc}  # add more items if needed
        
        values = self.pack_logging_values(values, test_prefix)
        self.log_dict(values, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['test', 'systematic_test',
                   'color_test', 'color_systematic_test',
                   'shape_test', 'shape_systematic_test',
                   'color_common_systematic_test', 'shape_common_systematic_test'][dataloader_idx]
        
        if self.config.multimodal_training:
            images, scenes, questions, labels = batch
            output = self.inner_model(images, scenes, questions)
        else:
            images, questions, labels = batch
            output = self.inner_model(images, questions)

        loss = F.cross_entropy(output, labels)

        pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(labels.view_as(pred)).sum().detach()
        count = torch.ones_like(pred).sum().detach()
        acc = correct / count

        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {"loss": loss, "acc": acc}  # add more items if needed
        
        values = self.pack_logging_values(values, test_prefix)
        self.log_dict(values, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.inner_model.parameters(), lr=self.config.lr)
        else:
            optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.config.lr)

        return optimizer

    def load_state_dict(self, state_dict, strict=True):
        # All this is quite hacky
        if isinstance(self.inner_model, Model):
            try:
                return super().load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                pos_size = self.config.n_patches + self.config.max_question_size
                state_dict['inner_model.pos_embedding'] = state_dict['inner_model.pos_embedding'][:pos_size,:,:]
                clf_state_dict = self.inner_model.classifier.state_dict()
                state_dict['inner_model.classifier.weight'] = clf_state_dict['weight']
                state_dict['inner_model.classifier.bias'] = clf_state_dict['bias']
                # word_embedding_state_dict = self.inner_model.word_embedding.state_dict()
                # print(word_embedding_state_dict['weight'].shape, state_dict['inner_model.word_embedding.weight'].shape)
                # word_embedding_state_dict['weight'][:-2] = state_dict['inner_model.word_embedding.weight']
                # state_dict['inner_model.word_embedding.weight'] = word_embedding_state_dict['weight']
                # state_dict['inner_model.word_embedding.bias'] = clf_state_dict['bias']

                return super().load_state_dict(state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)


class MultimodalPretrainingModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.inner_model = model
        self.save_hyperparameters()
        self.exp_prefix = ''

    def set_exp_prefix(self, prefix):
        self.exp_prefix = prefix

    def calc_accuracy(self, output, labels):
        pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        masked = labels != -100
        correct = pred.eq(labels.view_as(pred))[masked].sum().item()
        count = torch.ones_like(pred)[masked].sum().item()
        if not count:
            return 0.0
        return (correct / count)

    def calc_accuracy_at_perc(self, output, labels, percentage=0.1):
        masked = labels != -100
        topk = int(output.size(-1) * percentage)

        masked_output = output[masked]
        masked_labels = labels[masked]
        pred_sorted = torch.sort(masked_output, descending=True).indices  # get the index of the max log-probability

        correct = (pred_sorted[:,:topk] == masked_labels.unsqueeze(-1)).any(-1).sum().item()
        count = torch.ones_like(masked_labels).sum().item()
        if not count:
            return 0.0
        return correct / count

    def calc_rank(self, output, labels):
        pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        masked = labels != -100
        ordered_output = torch.sort(output, descending=True).indices
        ranks = (ordered_output == labels[:,:,None]).nonzero()[:,2] + 1
        return ranks.float().mean().item(), (1/ranks).mean().item()

    def calc_prob_metrics(self, output, labels):
        masked = labels != -100
        masked_output = output[masked].detach()
        masked_probs = F.softmax(masked_output, dim=-1)
        masked_labels = labels[masked]

        sorted_probs = torch.sort(masked_probs, descending=True).values
        top_prob = sorted_probs[:,0]
        gt_probs = masked_probs[torch.arange(masked_probs.size(0)), masked_labels]

        return gt_probs.mean().item(), (gt_probs / top_prob).mean().item()

    def pack_logging_values(self, metrics, prefix):
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def training_step(self, batch, batch_idx):
        images, scenes, labels = batch
        output = self.inner_model(images, scenes)
        loss = F.cross_entropy(output.transpose(1,2), labels)

        acc = self.calc_accuracy(output, labels)
        mr, mrr = self.calc_rank(output, labels)
        gt_prob, rel_gt_prob = self.calc_prob_metrics(output, labels)
        acc_at_5 = self.calc_accuracy_at_perc(output, labels, percentage=0.05)
        acc_at_10 = self.calc_accuracy_at_perc(output, labels, percentage=0.1)

        values = {"loss": loss, "acc": acc, "mr": mr, "mrr": mrr, "gt_prob": gt_prob, "rel_gt_prob": rel_gt_prob,
                  "acc@5%": acc_at_5, "acc@10%": acc_at_10}
        values = self.pack_logging_values(values, 'train')
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['val', 'systematic_val',
                   'color_val', 'color_systematic_val', 
                   'shape_val', 'shape_systematic_val', 
                   'color_common_systematic_val', 'shape_common_systematic_val'][dataloader_idx]
        images, scenes, labels = batch

        output = self.inner_model(images, scenes)
        loss = F.cross_entropy(output.transpose(1,2), labels)

        acc = self.calc_accuracy(output, labels)
        mr, mrr = self.calc_rank(output, labels)
        gt_prob, rel_gt_prob = self.calc_prob_metrics(output, labels)
        acc_at_5 = self.calc_accuracy_at_perc(output, labels, percentage=0.05)
        acc_at_10 = self.calc_accuracy_at_perc(output, labels, percentage=0.1)

        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {"loss": loss, "acc": acc, "mr": mr, "mrr": mrr, "gt_prob": gt_prob, "rel_gt_prob": rel_gt_prob,
                  "acc@5%": acc_at_5, "acc@10%": acc_at_10}

        values = self.pack_logging_values(values, test_prefix)
        self.log_dict(values, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        dl_name = ['test', 'systematic_test',
                   'color_test', 'color_systematic_test',
                   'shape_test', 'shape_systematic_test',
                   'color_common_systematic_test', 'shape_common_systematic_test'][dataloader_idx]
        images, scenes, labels = batch
        output = self.inner_model(images, scenes)
        loss = F.cross_entropy(output.transpose(1,2), labels)

        # pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        # masked = labels != -100
        # correct = pred.eq(labels.view_as(pred))[masked].sum().detach()
        # count = torch.ones_like(pred)[masked].sum().detach()
        acc = self.calc_accuracy(output, labels)
        mr, mrr = self.calc_rank(output, labels)
        gt_prob, rel_gt_prob = self.calc_prob_metrics(output, labels)
        acc_at_5 = self.calc_accuracy_at_perc(output, labels, percentage=0.05)
        acc_at_10 = self.calc_accuracy_at_perc(output, labels, percentage=0.1)

        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {"loss": loss, "acc": acc, "mr": mr, "mrr": mrr, "gt_prob": gt_prob, "rel_gt_prob": rel_gt_prob,
                  "acc@5%": acc_at_5, "acc@10%": acc_at_10}  # add more items if needed
        values = self.pack_logging_values(values, test_prefix)
        self.log_dict(values, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.config.optimizer == 'sgd':
            optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.config.lr)
        else:
            optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.config.lr)

        return optimizer


class ImageMaskingPretrainingModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.inner_model = model
        self.save_hyperparameters()
        self.exp_prefix = ''

    def set_exp_prefix(self, prefix):
        self.exp_prefix = prefix

    def _to_patches(self, image, flatten=True):
        patch_size = 16
        # Assumes batched
        n, c, h, w = image.shape
        np_h = h // patch_size
        np_w = w // patch_size

        y = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        y = y.contiguous().view(n, c, -1, patch_size, patch_size) # n,c,t,ph,pw
        y = y.permute(0, 2, 1, 3, 4) # n,c,t,ph,pw => n,t,c,ph,pw
        # y = y.permute(1, 0, 2, 3).contiguous()
        if flatten:
            y = y.flatten(2)

        return y

    def _to_image(self, patches):
        # Assumes batched
        patch_size = 16
        n = patches.size(0)
        c = 3
        np_h = np_w = int(patches.size(1)**0.5)

        x = patches.reshape(n, -1, c, patch_size, patch_size) # n,t,c,p,p
        x = x.permute(0, 2, 1, 3, 4).reshape(n, c, np_h, np_w, patch_size, patch_size) # n,t,c,p,p => n,c,t,p,p => n,c,np,np,p,p
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(n, c, patch_size*np_h, patch_size*np_w) # n,c,np,np,p,p => n,c,p,np,p,np =>
        return x

    def _sample_mask(self, patches, scene, p):
        n, tp, *_ = patches.shape
        n, ts, *_ = scene.shape
        t = tp + ts

        probability_matrix = torch.full((n, t), p)
        probability_matrix[:,tp+1:] = 0
        masked_combined = torch.bernoulli(probability_matrix).to(device=patches.device, dtype=torch.bool)
        return masked_combined[:,:tp], masked_combined

    def training_step(self, batch, batch_idx):
        image, scene = batch
        patches = self._to_patches(image)
        n_patches = patches.size(1)
        mask, masked_combined = self._sample_mask(patches, scene, p=self.config.mp_probability)
        assert masked_combined[:,n_patches+1:].sum() == 0, "Text token are being masked"

        output = self.inner_model(patches, scene, mask)
        output_patches = output[:,:n_patches,:]
        loss = F.mse_loss(output_patches, patches, reduction='none').mean(-1)
        # output = self._to_image(output_patches)

        loss.masked_fill_(~mask, value=0.0) # use only masked values for loss
        loss = loss.mean()

        # pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        # masked = labels != -100
        # correct = pred.eq(labels.view_as(pred))
        # correct = correct[masked].sum().detach()
        # count = torch.ones_like(pred)
        # count = count[masked].sum().detach()
        # acc = correct / count

        values = {
            "train_loss": loss,
            # "train_acc": acc
            }  # add more items if needed
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        image, scene = batch
        patches = self._to_patches(image)
        n_patches = patches.size(1)
        mask, masked_combined = self._sample_mask(patches, scene, p=self.config.mp_probability)
        assert masked_combined[:,n_patches+1:].sum() == 0, "Text token are being masked"

        output = self.inner_model(patches, scene, mask)
        output_patches = output[:,:n_patches,:]
        loss = F.mse_loss(output_patches, patches, reduction='none').mean(-1)
        # output = self._to_image(output_patches)

        loss.masked_fill_(mask, value=0.0)
        loss = loss.mean()

        # pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        # masked = labels != -100
        # correct = pred.eq(labels.view_as(pred))
        # correct = correct[masked].sum().detach()
        # count = torch.ones_like(pred)
        # count = count[masked].sum().detach()
        # acc = correct / count

        dl_name = ['val', 'systematic_val'][dataloader_idx]
        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {
            f"{test_prefix}_loss": loss,
            # "train_acc": acc
            }  # add more items if needed
        self.log_dict(values, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        image, scene = batch
        patches = self._to_patches(image)
        n_patches = patches.size(1)
        mask, masked_combined = self._sample_mask(patches, scene, p=self.config.mp_probability)
        assert masked_combined[:,n_patches+1:].sum() == 0, "Text token are being masked"

        output = self.inner_model(patches, scene, mask)
        output_patches = output[:,:n_patches,:]
        loss = F.mse_loss(output_patches, patches, reduction='none').mean(-1)
        # output = self._to_image(output_patches)

        loss.masked_fill_(mask, value=0.0)
        loss = loss.mean()

        # pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        # masked = labels != -100
        # correct = pred.eq(labels.view_as(pred))
        # correct = correct[masked].sum().detach()
        # count = torch.ones_like(pred)
        # count = count[masked].sum().detach()
        # acc = correct / count

        dl_name = ['val', 'systematic_val'][dataloader_idx]
        test_prefix = f'{self.exp_prefix}_{dl_name}' if self.exp_prefix else dl_name
        values = {
            f"{test_prefix}_loss": loss,
            # "train_acc": acc
            }  # add more items if needed
        self.log_dict(values, on_epoch=True, sync_dist=True)

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


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.n_tokens, config.d_hidden)
        if config.use_vit_embedding:
            vit_emb_size = 768
            vit_embedding = ViTEmbedding()
            self.patch_embedding = nn.Sequential(
                vit_embedding,
                nn.Linear(vit_emb_size, config.d_hidden),
            )
            if config.freeze_vit_embedding:
                for param in vit_embedding.parameters():
                    param.requires_grad = False

                vit_embedding.eval()
        elif config.use_embedding_loaded or config.use_vit_embedding_loaded:
            emb_size = 768 if config.use_vit_embedding_loaded else config.adapt_embedding_from

            self.patch_embedding = nn.Sequential(
                nn.Linear(emb_size, config.d_hidden)
                    if config.d_hidden != emb_size else nn.Identity()
            )
        else:
            patch_size = config.patch_height * config.patch_width * 3
            self.patch_embedding = nn.Sequential(
                Rearrange('b c  (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_height, p2=config.patch_width),
                nn.LayerNorm(patch_size),
                nn.Linear(patch_size, config.d_hidden),
                nn.LayerNorm(config.d_hidden),
            )

        if config.multimodal_pretraining:
            max_seq_len = config.n_patches + config.max_scene_size
        else:
            max_seq_len = config.n_patches + config.max_question_size
        # if config.multimodal_training:
        #     max_seq_len = max_seq_len + config.max_question_size
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, 1, config.d_hidden))
        self.type_image_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))
        self.classifier = nn.Linear(config.d_hidden, config.n_tokens)

    def train(self, mode=True):
        self = super().train(mode=mode)
        if self.config.use_vit_embedding and self.config.freeze_vit_embedding:
            vit_embedding = self.patch_embedding[0]
            vit_embedding.eval()
        return self

    def forward(self, image, text):
        embedded_text = self.word_embedding(text).transpose(1,0)
        embedded_image = self.patch_embedding(image).transpose(1,0)

        embedded_text = embedded_text + self.type_scene_embedding
        embedded_image = embedded_image + self.type_image_embedding
        combined_embedding = [embedded_image, embedded_text]

        i_pad_mask = torch.zeros_like(embedded_image[:,:,0], dtype=torch.bool).transpose(1,0)
        s_pad_mask = text == self.config.pad_idx
        pad_mask = [i_pad_mask, s_pad_mask]

        pad_mask = torch.cat(pad_mask, dim=1)
        combined_embedding = torch.cat(combined_embedding)

        transformer_input = combined_embedding + self.pos_embedding
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)

        output_logits = self.classifier(transformer_output).transpose(1,0)
        # if not self.training and self.aug_zero and self.aug_zero_independent:
        #     n, t, *_ = output_logits.shape
        #     output_logits.reshape(n, t, self.aug_zero+1, self.config.n_tokens).sum(-2)

        return output_logits


class MultimodalModelForImagePretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.aug_zero = config.aug_zero
        self.aug_zero_independent = config.aug_zero_independent
        n_input_tokens = config.n_tokens * (1 + self.aug_zero)

        patch_size = config.patch_height * config.patch_width * 3

        self.word_embedding = nn.Embedding(n_input_tokens, config.d_hidden)
        if config.use_vit_embedding:
            vit_emb_size = 768
            vit_embedding = ViTEmbedding()
            self.patch_embedding = nn.Sequential(
                vit_embedding,
                nn.Linear(vit_emb_size, config.d_hidden),
            )
            if config.freeze_vit_embedding:
                for param in vit_embedding.parameters():
                    param.requires_grad = False

                vit_embedding.eval()
        else:
            self.patch_embedding = nn.Linear(patch_size, config.d_hidden)
            # self.patch_embedding = nn.Sequential(
            #     Rearrange('b c  (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_height, p2=config.patch_width),
            #     nn.LayerNorm(patch_size),
            #     nn.Linear(patch_size, config.d_hidden),
            #     nn.LayerNorm(config.d_hidden),
            # )

        max_seq_len = config.n_patches + config.max_scene_size
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, 1, config.d_hidden))
        # if config.multimodal_training:
        #     max_seq_len = max_seq_len + config.max_question_size

        self.type_image_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.pred_patch_embedding = nn.Parameter(torch.randn(1, 1, patch_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))

        self.classifier = nn.Linear(config.d_hidden, patch_size)

    def train(self, mode=True):
        self = super().train(mode=mode)
        if self.config.use_vit_embedding and self.config.freeze_vit_embedding:
            vit_embedding = self.patch_embedding[0]
            vit_embedding.eval()
        return self

    def forward(self, patches, scene, mask):
        # if self.training and self.aug_zero:
        #     if self.aug_zero_independent:
        #         aug_zero_offset = torch.randint_like(scene, 0, self.aug_zero) * self.config.n_tokens
        #     else:
        #         aug_zero_offset = random.randint(0, self.aug_zero) * self.config.n_tokens
        # else:
        aug_zero_offset = 0

        mask = mask.unsqueeze(-1) # n,t -> n,t,1
        masked_patches = mask*self.pred_patch_embedding + ~mask*patches

        # if scene.dtype != torch.int64:
        #     scene = scene.long()
        embedded_scene = self.word_embedding(scene + aug_zero_offset).transpose(1,0)
        embedded_image = self.patch_embedding(masked_patches).transpose(1,0)

        embedded_scene = embedded_scene + self.type_scene_embedding
        embedded_image = embedded_image + self.type_image_embedding
        combined_embedding = [embedded_image, embedded_scene]

        i_pad_mask = torch.zeros_like(embedded_image[:,:,0], dtype=torch.bool).transpose(1,0)
        s_pad_mask = scene == self.config.pad_idx
        pad_mask = [i_pad_mask, s_pad_mask]

        pad_mask = torch.cat(pad_mask, dim=1)
        combined_embedding = torch.cat(combined_embedding)

        transformer_input = combined_embedding + self.pos_embedding
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        # if self.config.multimodal_training:
        #     return self.classifier(transformer_output[0,:,:])
        # else:
        output_logits = self.classifier(transformer_output).transpose(1,0)
        # if not self.training and self.aug_zero and self.aug_zero_independent:
        #     n, t, *_ = output_logits.shape
        #     output_logits.reshape(n, t, self.aug_zero+1, self.config.n_tokens).sum(-2)

        return output_logits


class ViTEmbedding(nn.Module):
    def __init__(self, checkpoint_to_use='google/vit-base-patch16-224'):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(checkpoint_to_use)
        self.vit = ViTModel.from_pretrained(checkpoint_to_use, add_pooling_layer=False)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.vit(**inputs)
        return outputs['last_hidden_state']

    @property
    def device(self):
        return next(self.parameters()).device


class MaskedImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.aug_zero = config.aug_zero
        self.aug_zero_independent = config.aug_zero_independent
        n_tokens = config.n_tokens * (1 + self.aug_zero)

        patch_size = config.patch_height * config.patch_width * 3
        self.word_embedding = nn.Embedding(n_tokens, config.d_hidden)
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3,
        #               config.d_hidden,
        #               kernel_size=(config.patch_height, config.patch_width),
        #               stride=(config.patch_height, config.patch_width)),
        #     Rearrange('b d hp wp -> b (hp wp) d'),
        # )
        self.patch_embedding = nn.Linear(patch_size, config.d_hidden)

        max_seq_len = config.n_patches + config.max_scene_size
        if config.multimodal_training:
            max_seq_len = max_seq_len + config.max_question_size
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, 1, config.d_hidden))
        self.type_image_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))

        # if config.masked_image_pretraining:
            # self.type_question_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.classifier = nn.Linear(config.d_hidden, patch_size)
        # elif config.multimodal_training:
        #     self.type_question_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        #     self.classifier = nn.Linear(config.d_hidden, config.n_outputs)
        # else:
        #     self.classifier = nn.Linear(config.d_hidden, n_tokens)

    def forward(self, patches, scene, question=None, masked_patches=None):
        if self.training and self.aug_zero:
            if self.aug_zero_independent:
                aug_zero_offset_scene = torch.randint_like(scene, 0, self.aug_zero) * self.config.n_tokens
                if question is not None:
                    aug_zero_offset_question = torch.randint_like(question, 0, self.aug_zero) * self.config.n_tokens
            else:
                aug_zero_offset_scene = aug_zero_offset_question = random.randint(0, self.aug_zero) * self.config.n_tokens
        else:
            aug_zero_offset_scene = aug_zero_offset_question = 0

        embedded_scene = self.word_embedding(scene + aug_zero_offset_scene).transpose(1,0)
        embedded_image = self.patch_embedding(patches).transpose(1,0)

        embedded_scene = embedded_scene + self.type_scene_embedding
        embedded_image = embedded_image + self.type_image_embedding
        combined_embedding = [embedded_image, embedded_scene]

        i_pad_mask = torch.zeros_like(embedded_image[:,:,0], dtype=torch.bool).transpose(1,0)
        s_pad_mask = scene == self.config.pad_idx
        pad_mask = [i_pad_mask, s_pad_mask]

        if question is not None and self.config.multimodal_training:
            embedded_question = self.word_embedding(question + aug_zero_offset_question).transpose(1,0)
            embedded_question = embedded_question + self.type_question_embedding
            q_pad_mask = question == self.config.pad_idx
            pad_mask.append(q_pad_mask)
            combined_embedding.append(embedded_question)

        pad_mask = torch.cat(pad_mask, dim=1)
        combined_embedding = torch.cat(combined_embedding)

        transformer_input = combined_embedding + self.pos_embedding

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        # if self.config.masked_image_pretraining:
        return rearrange(self.classifier(transformer_output), '(hp wp) b d -> b d hp wp')
        # elif self.config.multimodal_training:
        #     return self.classifier(transformer_output[0,:,:])
        # else:
        #     return self.classifier(transformer_output).transpose(1,0)


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
            if self.aug_zero_independent:
                aug_zero_offset_scene = torch.randint_like(scene, 0, self.aug_zero) * self.config.n_tokens
                if question is not None:
                    aug_zero_offset_question = torch.randint_like(question, 0, self.aug_zero) * self.config.n_tokens
            else:
                aug_zero_offset_scene = aug_zero_offset_question = random.randint(0, self.aug_zero) * self.config.n_tokens
        else:
            aug_zero_offset_scene = aug_zero_offset_question = 0

        embedded_scene = self.word_embedding(scene + aug_zero_offset_scene).transpose(1,0)
        embedded_question = self.word_embedding(question + aug_zero_offset_question).transpose(1,0)

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
        pad_mask = torch.cat((q_pad_mask, i_pad_mask), dim=1)

        combined_embedding = torch.cat((embedded_question, embedded_image))
        transformer_input = combined_embedding + self.pos_embedding

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=pad_mask)
        return self.classifier(transformer_output[0,:,:])

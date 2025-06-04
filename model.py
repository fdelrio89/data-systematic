import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import lightning as L


def build_model(config):
    print("Creating MultimodalModel model")
    model = MultimodalModel(config)

    print("Creating MultimodalPretrainingModel training model")
    training_model = MultimodalPretrainingModel(model, config)

    if config.start_from:
        resume_from_path = f"outputs/{config.start_from}/last.ckpt"
        training_model.load_state_dict(torch.load(resume_from_path)['state_dict'])

    return training_model


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


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.n_tokens, config.d_hidden)
        patch_size = config.patch_height * config.patch_width * 3
        self.patch_embedding = nn.Sequential(
            Rearrange('b c  (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.patch_height, p2=config.patch_width),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, config.d_hidden),
            nn.LayerNorm(config.d_hidden),
        )

        max_seq_len = config.n_patches + config.max_scene_size

        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, 1, config.d_hidden))
        self.type_image_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))
        self.type_scene_embedding = nn.Parameter(torch.randn(1, 1, config.d_hidden))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_hidden, nhead=config.n_head, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers, norm=nn.LayerNorm(config.d_hidden))
        self.classifier = nn.Linear(config.d_hidden, config.n_tokens)

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

        return output_logits


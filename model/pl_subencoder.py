from einops import rearrange
from functools import partial

import numpy as np
import torch

from transformers import AutoModel, T5EncoderModel, AutoConfig, get_linear_schedule_with_warmup
import pytorch_lightning as pl

from model.layers import Pooling, MLPLayer
from model.losses import LOSS_CLASSES

from utils.data_utils import gather_with_var_len


class LitSubEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()  # This will save all arguments / params passed to __init__()

        self.params = params

        # Backbone encoder (from huggingface)
        if "t5" in params.model_name:
            self.encoder = T5EncoderModel.from_pretrained(params.model_name)
        else:
            self.encoder = AutoModel.from_pretrained(params.model_name)

        self.encoder_config = AutoConfig.from_pretrained(params.model_name)
        self.encoder_hidden_dim = self.encoder_config.hidden_size

        # if mlp hidden dim is None, set it to match the encoder hidden dim
        self.mlp_hidden_dim = self.encoder_hidden_dim if params.mlp_hidden_dim is None else params.mlp_hidden_dim
        self.final_output_dim = self.encoder_hidden_dim if params.final_output_dim is None else params.final_output_dim

        # Pooling
        self.pooler = Pooling()

        # MLP layer
        self.mlp = MLPLayer(input_dim=self.encoder_hidden_dim,
                            hidden_dim=self.mlp_hidden_dim,
                            output_dim=self.final_output_dim)

        # Loss function
        self.loss = LOSS_CLASSES[params.loss_type]()


    def forward(self,
                encoder_inputs,
                num_prop_per_sentence,
                all_prop_mask,
                pool_all_tokens: bool = False):
        """

        :param encoder_inputs: inputs for encoder
        :param num_prop_per_sentence:
        :param all_prop_mask:
        :param prop_labels:
        :return:
        """

        outputs = self.encoder(**encoder_inputs)
        encoded = outputs["last_hidden_state"]

        # repeat each encoded sentence by number of propositions in the sentence
        encoded_repeat = encoded.repeat_interleave(num_prop_per_sentence, dim=0)

        # Pool the encoded representations with respect to the proposition masks
        if pool_all_tokens:
            all_tok_attn_repeat = encoder_inputs.attention_mask.repeat_interleave(num_prop_per_sentence, dim=0)
            prop_pooled = self.pooler(token_embeddings=encoded_repeat,
                                      attention_mask=all_tok_attn_repeat)
        else:
            prop_pooled = self.pooler(token_embeddings=encoded_repeat,
                                      attention_mask=all_prop_mask)

        # Pass the pooled representation through a MLP layer
        prop_pooled = self.mlp(prop_pooled)

        return prop_pooled

    def configure_optimizers(self):

        # TODO(sihaoc): maybe add learning rate scaling
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.params.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    #####
    # Defines how each forward step
    #####
    def _step(self, batch):

        props_encoded = self.forward(
            encoder_inputs=batch.encoder_inputs,
            num_prop_per_sentence=batch.num_prop_per_sentence,
            all_prop_mask=batch.all_prop_mask
        )

        return {
            "embeddings": props_encoded,
            "labels": batch.prop_labels,
            "num_prop_per_sentence": batch.num_prop_per_sentence
        }

    def training_step(self, train_batch, batch_idx):
        shard_output = self._step(train_batch)

        # With DDP, each GPU gets its own mini-batch.
        # Since the labels are created by enumerating instances within batch, i.e. labels ranges from 0 to batch_size
        # within each batch, when merging the batches, we need to differentiate between the labels of each batch
        # This is handled by gather_with_var_len
        gather_fn = partial(self.all_gather, sync_grads=True)
        all_output = {
            key: gather_with_var_len(val,
                                     base_gather_fn=gather_fn,
                                     add_label_offset=(key == 'labels')) for key, val in shard_output.items()
        }

        # As all_output after gather() will create an extra dimension for # GPUs before the batch dimension
        # We flatten the # GPU and batch dimensions
        prop_embeddings = all_output["embeddings"]
        labels = all_output["labels"]
        num_prop_per_sentence = all_output["num_prop_per_sentence"]

        loss = self.loss(
            prop_embeddings,
            labels
        )

        self.log("loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=num_prop_per_sentence.size(0), sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        shard_output = self._step(val_batch)
        gather_fn = partial(self.all_gather, sync_grads=True)
        all_output = {
            key: gather_with_var_len(val,
                                     base_gather_fn=gather_fn,
                                     add_label_offset=(key == 'labels')) for key, val in shard_output.items()
        }

        prop_embeddings = all_output["embeddings"]
        labels = all_output["labels"]

        with torch.no_grad():
            loss = self.loss(
                prop_embeddings,
                labels
            )
        return loss.item()

    def validation_epoch_end(self, outputs) -> None:
        # take the mean of the loss
        val_loss = np.mean(outputs)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch)

#
# Copyright 2026 David Ragnar Nelson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Main entrypoint for deploying TrOCR models
"""
import dataclasses

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchmetrics.text import CharErrorRate, WordErrorRate

from PIL import Image, ImageOps

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dotenv import load_dotenv
import os

from .utils import rotate_segment


load_dotenv()
token = str(os.getenv('HF_TOKEN'))

############
# Training
###########

# torch Dataset to represent training data
class TrOCRTrainingDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name and text
        file_name = self.df['image'][idx]
        text = self.df['text'][idx]
        # image processing
        image = Image.open(os.path.join(self.root_dir, file_name)).convert('RGB')
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding='max_length',
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {'pixel_values': pixel_values.squeeze(), 'labels': torch.tensor(labels)}
        return encoding


class TrOCRModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset=None, eval_dataset=None):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.train_dataset = None
        if train_dataset:
            self.train_dataset = train_dataset

        self.eval_dataset = None
        if eval_dataset:
            self.eval_dataset = eval_dataset

        self.batch_size = config.get('batch_size')
        self.lr = self.config.get('lr')

        self.val_cer = CharErrorRate()
        # self.val_wer = WordErrorRate()

        self.save_hyperparameters()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model.generate(batch, output_scores=True, return_dict_in_generate=True)
        pred = processor.batch_decode(outputs['sequences'], skip_special_tokens=True)
        return pred, outputs['sequences_scores']

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        outputs = self.model.generate(batch['pixel_values'])
        pred = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = batch['labels']
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        labels = processor.batch_decode(labels, skip_special_tokens=True)

        self.val_cer.update(pred, labels)
        # self.val_wer.update(outputs, batch['labels'])

    def on_validation_epoch_end(self):
        epoch_cer = self.val_cer.compute()
        self.log('val_accuracy', epoch_cer, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_word_accuracy', self.val_wer, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # reset CER and WER
        self.val_cer.reset()
        # self.val_wer.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.model.generate(batch['pixel_values'])
        pred = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = batch['labels']
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        labels = processor.batch_decode(labels, skip_special_tokens=True)

        self.val_cer.update(pred, labels)

    def on_test_epoch_end(self):
        epoch_cer = self.val_cer.compute()
        self.log('test_cer', epoch_cer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get('lr'))
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


def save_model_to_safetensors(ckpt, path):
    from safetensors.torch import save_model
    model = TrOCRModel.load_from_checkpoint(ckpt)
    save_model(model.state_dict(), path)


#############
# Inference
############


class TrOCRInferenceDataset(Dataset):
    '''Dataset to run inference on pre-segmented pages using TrOCR'''
    def __init__(self, image, lines, processor, flip=False, max_target_length=128):
        self.image = image
        self.lines = lines
        self.processor = processor
        self.flip = flip
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        im = Image.open(self.image)
        crop, im = rotate_segment(im, self.lines[idx])
        im = im.crop(crop)
        if self.flip:
            im = ImageOps.flip(im)
            im = ImageOps.mirror(im)
        pixel_values = self.processor(im, return_tensors='pt').pixel_values

        encoding = pixel_values.squeeze()
        return encoding


def load_model(processor, model):
    if processor:
        processor = TrOCRProcessor.from_pretrained(processor, token=token)
    else:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', token=token)

    if model:
        model = VisionEncoderDecoderModel.from_pretrained(model, token=token)
    else:
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten', token=token)

    return processor, model


def predict(
    image,
    lines,
    processor=None,
    model=None,
    batch_size=8,
    num_workers=4
):
    seg_len = len(lines)
    idx = 0

    # prepare data
    inference_dataset = TrOCRInferenceDataset(
        image=image,
        lines=lines,
        processor=processor,
    )

    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    results = [None] * seg_len
    confidences = [None] * seg_len
    bad_lines = []
    for batch in dataloader:
        output = model.generate(batch, output_scores=True, return_dict_in_generate=True)
        logits, scores = output['sequences'], output['sequences_scores']
        preds = processor.batch_decode(logits, skip_special_tokens=True)
        for pred, score in zip(preds, scores):
            # keep track of lines with unacceptable confidences
            if score < -0.05:
                bad_lines.append(idx)
            results[idx] = pred
            confidences[idx] = score
            idx = idx + 1

    bad_dataset = TrOCRInferenceDataset(
        image=image,
        lines=[lines[i] for i in bad_lines],
        processor=processor,
        flip=True,
    )

    bad_dataloader = DataLoader(
        bad_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    n = 0
    for batch in bad_dataloader:
        output= model.generate(batch, output_scores=True, return_dict_in_generate=True)
        logits, scores = output['sequences'], output['sequences_scores']
        preds = processor.batch_decode(logits, skip_special_tokens=True)
        for pred, score in zip(preds, scores):
            first_score = confidences[bad_lines[n]]
            # if confidence is better, keep the new prediction, otherwise go with the first one
            if score > first_score:
                results[bad_lines[n]] = pred
                confidences[bad_lines[n]] = scores
            n = n + 1

    return results, confidences

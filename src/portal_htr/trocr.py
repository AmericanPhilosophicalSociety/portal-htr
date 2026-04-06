#
# Copyright 2026 David Ragnar Nelson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entrypoint for deploying TrOCR models
"""
import dataclasses

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchmetrics.text import CharErrorRate, WordErrorRate

from PIL import Image, ImageFile

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from kraken.lib.segmentation import extract_polygons
from kraken.containers import BaselineOCRRecord

from dotenv import load_dotenv
import os


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
        outputs = self.model.generate(batch)
        pred = processor.batch_decode(outputs, skip_special_tokens=True)
        return pred

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

# adapted from kraken.lib.vgsl.rpred
def _extract_line(im, segmentation, line_idx, legacy: bool = False):
    '''
    Given an image, segmentation, and line index, return a single
    extracted line as a PIL Image
    '''

    line = segmentation.lines[line_idx]
    seg = dataclasses.replace(segmentation, lines=[line])
    # try:
        # im, _ = next(extract_polygons(im, seg, legacy=legacy))
        # return im, line_idx
    # except ValueError:
        # return None, line_idx
    im, _ = next(extract_polygons(im, seg, legacy=legacy))
    return im, line_idx

class TrOCRInferenceDataset(Dataset):
    '''Dataset to run inference on pre-segmented pages using TrOCR'''
    def __init__(self, image, lines, processor, max_target_length=128):
        self.image = image
        self.lines = lines
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.lines.lines)

    def __getitem__(self, idx):
        im = Image.open(self.image)
        im, _ = _extract_line(im, self.lines, idx)
        pixel_values = self.processor(im, return_tensors='pt').pixel_values

        encoding = pixel_values.squeeze()
        return encoding


def predict(
    image,
    segmentation,
    processor=None,
    model=None,
    batch_size=8,
    num_workers=4
):
    seg_len = len(segmentation.lines)
    # rec_results= [None] * seg_len
    idx = 0

        # load processor and model
    if processor:
        processor = TrOCRProcessor.from_pretrained(processor, token=token)
    else:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', token=token)

    if model:
        model = VisionEncoderDecoderModel.from_pretrained(model, token=token)
    else:
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten', token=token)

    # prepare data
    inference_dataset = TrOCRInferenceDataset(
        image=image,
        lines=segmentation,
        processor=processor,
    )

    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    results = []
    for batch in dataloader:
        logits = model.generate(batch)
        preds = processor.batch_decode(logits, skip_special_tokens=True)
        for pred in preds:
            rec = _recognize_lines(pred, idx, segmentation)
            results.append(rec)
            # rec_results[idx] = rec
            # idx = idx + 1

    return results


def _recognize_lines(pred, idx, segmentation):
    line = segmentation.lines[idx]
    record = BaselineOCRRecord(pred, line.boundary, [], line)
    return record


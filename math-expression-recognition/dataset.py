import os
import random
from typing import Dict, Tuple, List
from overrides import overrides

import cv2
import numpy as np
import pandas as pd

import torch

import spacy

import allennlp

from allennlp.common.util import START_SYMBOL, END_SYMBOL, get_spacy_model

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer, WordTokenizer

SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv
def resize(img, size):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT)

    return scaled_img

class MathTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    # Tokenize latex
    # From https://codereview.stackexchange.com/questions/186024/basic-equation-tokenizer
    def _split(self, s):
        out = []
        buf = ''
        for l in s:
            if not l.isalnum():
                if buf:
                    out += [buf]
                    buf = ''
                out += [l]
            else:
                if l.isalpha() and buf.isdigit() or l.isdigit() and buf.isalpha():
                    out += [buf]
                    buf = ''
                buf += l
        if buf:
            out += [buf]
            
        out = [Token(char) for char in out]
        return out

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = self._split(text)

        return tokens

@DatasetReader.register('math-dataset')
class MathDatasetReader(DatasetReader):
    def __init__(self, root_path: str, size: int = 512, lazy: bool = True, subset: bool = False) -> None:
        super().__init__(lazy)
        
        self.mean = 0.4023
        self.std = 0.4864
        
        self.root_path = root_path
        self.size = size
        self.subset = subset
        
        self._tokenizer = MathTokenizer()
        self._token_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file: str):
        df = pd.read_csv(os.path.join(self.root_path, file))
        if self.subset:
            df = df.loc[:2]

        for _, row in df.iterrows():
            img_id = row['id']
            
            if 'label' in df.columns:
                label = row['label']
                yield self.text_to_instance(file, img_id, label)
            else:
                yield self.text_to_instance(file, img_id)
            
    @overrides
    def text_to_instance(self, file: str, img_id: int, label: str = None) -> Instance:
        sub_path = 'test' if file == 'test.csv' else 'train'
        path = os.path.join(self.root_path, sub_path, f'{img_id}.png')
        
        img = cv2.imread(path)
        img = img / 255
        img = resize(img, (self.size, self.size))
        img = (img - self.mean) / self.std
        img = img.reshape(3, self.size, self.size)
        
        fields = {}
        fields['img'] = ArrayField(img)
        
        if label is not None:
            label = self._tokenizer.tokenize(label)

            label.insert(0, Token(START_SYMBOL))
            label.append(Token(END_SYMBOL))
            
            fields['label'] = TextField(label, self._token_indexer)
        
        return Instance(fields)

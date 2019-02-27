import os
import random
from typing import Dict, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import allennlp

from allennlp.common import Registrable, Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.token_embedders import Embedding

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.beam_search import BeamSearch

from allennlp.training.metrics import F1Measure, BLEU

from math_handwriting_recognition.metrics import Exprate

SEED = 42

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageCaptioningAttention(nn.Module, Registrable):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int, doubly_stochastic_attention: bool = True) -> None:
        super().__init__()
                
        self._encoder_dim = encoder_dim
        self._decoder_dim = decoder_dim
        self._attention_dim = attention_dim
        
        self._doubly_stochastic_attention = doubly_stochastic_attention
        
        self._encoder_attention = nn.Linear(self._encoder_dim, self._attention_dim)
        self._decoder_attention = nn.Linear(self._decoder_dim, self._attention_dim)
        self._attention = nn.Linear(self._attention_dim, 1)
        
        if self._doubly_stochastic_attention:
            self._f_beta = nn.Linear(self._decoder_dim, self._encoder_dim)
        
    @overrides
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: (batch_size, height * width, attention_dim)
        encoder_attention = self._encoder_attention(x)
        # Shape: (batch_size, 1, attention_dim)
        decoder_attention = self._decoder_attention(h).unsqueeze(1)
                
        # Shape: (batch_size, height * width)
        attention = self._attention(F.relu((encoder_attention + decoder_attention), inplace=True)).squeeze(2)
        
        # No need for masked softmax since all encoder pixels are available and hidden state of rnn isn't masked
        # Shape: (batch_size, h * w, 1)
        attention_weights = torch.softmax(attention, dim=1).unsqueeze(2)

        # Shape: (batch_size, encoder_dim)
        attention = (x * attention_weights).sum(dim=1)
        
        if self._doubly_stochastic_attention:     
            # Shape: (batch_size, encoder_dim)
            gate = torch.sigmoid(self._f_beta(h))
            # Shape: (batch_size, encoder_dim)
            attention = gate * attention
        
        return attention, attention_weights
    
    def get_output_dim(self) -> int:
        return self._encoder_dim

class ImageCaptioningDecoder(nn.Module):
    def __init__(self, vocab_size:int = 70, encoder_dim:int = 512, embedding_dim:int = 64, attention_dim:int = 64, decoder_dim:int = 64):
        super(ImageCaptioningDecoder, self).__init__()
        
        self._vocab_size = vocab_size
        self._encoder_dim = encoder_dim
        self._embedding_dim = embedding_dim
        self._attention_dim = attention_dim
        self._decoder_dim = decoder_dim
        
        self._embedding = Embedding(self._vocab_size, self._embedding_dim)
        self._attention = ImageCaptioningAttention(self._encoder_dim, self._decoder_dim, self._attention_dim)
        self._decoder_cell = nn.LSTMCell(self._embedding.get_output_dim() + self._attention.get_output_dim(), self._decoder_dim)
        self._linear = nn.Linear(self._decoder_dim, self._vocab_size)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, predicted_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self._embedding(predicted_indices).float().view(-1, self._embedding_dim) # (batch_size, embedding_dim) (64, 64)

        # Shape: (batch_size, encoder_dim) (batch_size, h * w, 1)
        attention, attention_weights = self._attention(x, h)

        ## Change to not use teacher forcing all the time
        # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim)
        h, c = self._decoder_cell(torch.cat([attention, embedding], dim=1), (h, c))
        
        # Get output predictions (one per character in vocab)
        # Shape: (batch_size, vocab_size)
        preds = self._linear(h)

        return h, c, preds, attention_weights
    
@Model.register('math-image-captioning')
class ImageCaptioning(Model):
    def __init__(self, vocab: Vocabulary, max_timesteps: int = 50, encoder_size: int = 14, encoder_dim: int = 512, 
                 embedding_dim: int = 64, attention_dim: int = 64, decoder_dim: int = 64, beam_size: int = 3, teacher_forcing: bool = True) -> None:
        super().__init__(vocab)
        
        self._max_timesteps = max_timesteps
        
        self._vocab_size = self.vocab.get_vocab_size()
        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        # POSSIBLE CHANGE LATER
        self._pad_index = self.vocab.get_token_index('@@PADDING@@')
        
        self._encoder_size = encoder_size
        self._encoder_dim = encoder_dim
        self._embedding_dim = embedding_dim
        self._attention_dim = attention_dim
        self._decoder_dim = decoder_dim
        
        self._beam_size = beam_size
        self._teacher_forcing = teacher_forcing

        self._init_h = nn.Linear(self._encoder_dim, self._decoder_dim)
        self._init_c = nn.Linear(self._encoder_dim, self._decoder_dim)
        
        self._resnet = torchvision.models.resnet18()
        modules = list(self._resnet.children())[:-2]
        self._encoder = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d(self._encoder_size)
        )

        self._decoder = ImageCaptioningDecoder(self._vocab_size, self._encoder_dim, self._embedding_dim, self._attention_dim, self._decoder_dim)
        
        self.beam_search = BeamSearch(self._end_index, self._max_timesteps, self._beam_size)
        
        self._bleu = BLEU(exclude_indices={self._start_index, self._end_index, self._pad_index})
        self._exprate = Exprate(self._end_index)

    def _init_hidden(self, encoder: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_encoder = encoder.mean(dim=1)
        
        # Shape: (batch_size, decoder_dim)
        initial_h = self._init_h(mean_encoder)
        # Shape: (batch_size, decoder_dim)
        initial_c = self._init_c(mean_encoder)

        return initial_h, initial_c
    
    def _decode(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        x = state['x']
        h = state['h']
        c = state['c']
        label = state['label']
        mask = state['mask']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Sort data to be able to only compute relevent parts of the batch at each timestep
        # Shape: (batch_size)
        lengths = mask.sum(dim=1)
        # Shape: (batch_size) (batch_size)
        sorted_lengths, indices = lengths.sort(dim=0, descending=True)
        # Computing last timestep isn't necessary with labels since last timestep is eos token or pad token 
        timesteps = sorted_lengths[0] - 1

        # Shape: (batch_size, height * width, encoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, decoder_dim)
        # Shape: (batch_size, timesteps)
        # Shape: (batch_size, timesteps)
        x = x[indices]
        h = h[indices]
        c = c[indices]
        label = label[indices]        
        mask = mask[indices]
        
        # Shape: (batch_size, 1)
        predicted_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1, 1)
        
        # Shape: (batch_size, timesteps, vocab_size)
        predictions = torch.zeros(local_batch_size, timesteps, self._vocab_size, device=device)
        attention_weights = torch.zeros(local_batch_size, timesteps, self._encoder_size * self._encoder_size, device=device)
        
        for t in range(timesteps):
            # Shape: (batch_offset)
            batch_offset = sum([l > t for l in sorted_lengths.tolist()])

            # Only compute data in valid timesteps
            # Shape: (batch_offset, height * width, encoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, decoder_dim)
            # Shape: (batch_offset, 1)
            x_t = x[:batch_offset]
            h_t = h[:batch_offset]
            c_t = c[:batch_offset]
            predicted_indices_t = predicted_indices[:batch_offset]
            
            # Decode timestep
            # Shape: (batch_size, decoder_dim) (batch_size, decoder_dim) (batch_size, vocab_size), (batch_size, encoder_dim, 1)
            h, c, preds, attention_weight = self._decoder(x_t, h_t, c_t, predicted_indices_t)
            
            # Get new predicted indices to pass into model at next timestep
            # Use teacher forcing if chosen
            if self._teacher_forcing:
                # Send next timestep's label to next timestep
                # Shape: (batch_size, 1)
                predicted_indices = label[:batch_offset, t + 1].view(-1, 1)
            else:
                # Shape: (batch_size, 1)
                predicted_indices = torch.argmax(preds, dim=1).view(-1, 1)
            
            # Save preds
            predictions[:batch_offset, t, :] = preds
            attention_weights[:batch_offset, t, :] = attention_weight.view(-1, self._encoder_size * self._encoder_size)
            
        # Update state and add logits
        state['x'] = x
        state['h'] = h
        state['c'] = c
        state['label'] = label
        state['mask'] = mask
        state['attention_weights'] = attention_weights
        state['logits'] = predictions
            
        return state
    
    def _beam_search_step(self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Group_size is batch_size * beam_size except for first decoding timestep where it is batch_size
        # Shape: (group_size, decoder_dim) (group_size, decoder_dim) (group_size, vocab_size)
        h, c, predictions, _ = self._decoder(state['x'], state['h'], state['c'], last_predictions)

        # Update state
        # Shape: (group_size, decoder_dim)
        state['h'] = h
        # Shape: (group_size, decoder_dim)
        state['c'] = c
        
        # Run log_softmax over logit predictions
        # Shape: (group_size, vocab_size)
        log_preds = F.log_softmax(predictions, dim=1)

        return log_preds, state
    
    def _beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get data from state
        x = state['x']
        h = state['h']
        c = state['c']
        
        # Get actual size of current batch
        local_batch_size = x.shape[0]

        # Beam search wants initial preds of shape: (batch_size)
        # Shape: (batch_size)
        initial_indices = torch.LongTensor([[self._start_index]] * local_batch_size).to(device).view(-1)
        
        state = {'x': x, 'h': h, 'c': c}
        
        # Timesteps returned aren't necessarily max_timesteps
        # Shape: (batch_size, beam_size, timesteps), (batch_size, beam_size)
        predictions, log_probabilities = self.beam_search.search(initial_indices, state, self._beam_search_step)
        
        # Only keep best predictions from beam search
        # Shape: (batch_size, timesteps)
        predictions = predictions[:, 0, :].view(local_batch_size, -1)
        
        return predictions
        
    @overrides
    def forward(self, img: torch.Tensor, label: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Encode the image
        # Shape: (batch_size, encoder_dim, height, width)
        x = self._encoder(img)
        
        # Flatten image
        # Shape: (batch_size, height * width, encoder_dim)
        x = x.view(x.shape[0], -1, x.shape[1])

        state = {'x': x}
        # Compute loss on train and val
        if label is not None:
            # Initialize h and c
            # Shape: (batch_size, decoder_dim)
            state['h'], state['c'] = self._init_hidden(x)
            
            # Convert label dict to tensor since label isn't an input to the model and get mask
            # Shape: (batch_size, timesteps)
            state['mask'] = get_text_field_mask(label).to(device)
            # Shape: (batch_size, timesteps)
            state['label'] = label['tokens']

            # Decode encoded image and get loss on train and val
            state = self._decode(state)

            # Loss shouldn't be computed on start token
            state['mask'] = state['mask'][:, 1:].contiguous()
            state['target'] = state['label'][:, 1:].contiguous()
            
            # Compute cross entropy loss
            state['loss'] = sequence_cross_entropy_with_logits(state['logits'], state['target'], state['mask'])
            # Doubly stochastic regularization
            state['loss'] += ((1 - torch.sum(state['attention_weights'], dim=1)) ** 2).mean()

        # Decode encoded image with beam search on val and test
        if not self.training:
            # (Re)initialize h and c
            state['h'], state['c'] = self._init_hidden(x)
            
            # Run beam search
            state['out'] = self._beam_search(state)
            
            # Compute validation scores
            if 'label' in state:
                self._bleu(state['out'], state['target'])
                self._exprate(state['out'], state['target'])
            
        # Set out to logits while training
        else:
            state['out'] = state['logits']
            
        # Create output_dict
        output_dict = {}
        output_dict['out'] =  state['logits'] if self.training else state['out']
        
        if 'loss' in state:
            output_dict['loss'] = state['loss']

        return output_dict
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Return Bleu score if possible
        if not self.training:
            metrics.update(self._bleu.get_metric(reset))
            metrics.update(self._exprate.get_metric(reset))
            
        return metrics
        
    def _trim_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        for b in range(predictions.shape[0]):
            # Shape: (timesteps)
            predicted_index = predictions[b]
            # Set last predicted index to eos token in case there are no predicted eos tokens
            predicted_index[-1] = self._end_index

            # Get index of first eos token
            # Shape: (timesteps)
            mask = predicted_index == self._end_index
            # Work around for pytorch not having an easy way to get the first non-zero index
            eos_token_idx = list(mask.cpu().numpy()).index(1)
            
            # Set prediction at eos token's timestep to eos token
            predictions[b, eos_token_idx] = self._end_index
            # Replace all timesteps after first eos token with pad token
            predictions[b, eos_token_idx + 1:] = self._pad_index

        return predictions

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Trim test preds to first eos token
        # Shape: (batch_size, timesteps)
        output_dict['out'] = self._trim_predictions(output_dict['out'])

        return output_dict

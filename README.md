# math-expression-recognition

## Table Of Contents

[edit][1]

[1]: https://github.com/bkahn-github/math-expression-recognition/edit/master/README.md

## ToDo

Look into all math recognition papers

Use more dimensions

Increase the batch size

Larger/pretrained resnet

Preprocess latex like https://github.com/harvardnlp/im2markup

Use im2latex dataset for pretraining http://lstm.seas.harvard.edu/latex/

# Transformers

https://github.com/ruotianluo/Transformer_Captioning

https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html

> Paper: http://aclweb.org/anthology/P18-1238

https://openreview.net/forum?id=HJN6DiAcKQ

http://openaccess.thecvf.com/content_ICCV_2017/papers/Pedersoli_Areas_of_Attention_ICCV_2017_paper.pdf

https://www.researchgate.net/publication/325016817_Captioning_Transformer_with_Stacked_Attention_Modules

## Later

fp16 **Wait until allennlp supports apex**

## Done

Expose jupyter port through pf service

Document exprate

Specify correct validation metric

Use a learning rate scheduler

Change tokenizer to count \sum as a single token

make sure timestep t is indexed correctly

turn off teacher forcing

Don't compute loss on start/end tokens? for train and val

Implement val loss and scores

Check other image captioning repositories 

Check if doubly stochastic attention regularization should be used

beam search and bleu score should be in forward()

Move comments out of inline

CHANGE code to use state more?

Use get_output_dim more often

Fix train-val split 

Change code to use config files 

Check all of the allennlp cli commands 

Change decode to only keep relevant data in state dict

Prepend all private instance variables with _

Try allennlp's embedding Done

## Other

Scheduled sampling

Make sure label's mask isn't nescessary for validation loss and score (validation is same as training but without backwards())

CHECK IF VALIDATION SHOULD USE TEACHER FORCING OR NOT NOT; Look at beam search part below (it should be run at validation)

validation loss should be computed with decode_train (Not relevant)

CHECK masking before softmax (Not nescessary right now but will keep in mind for later)

BEAM SEARCH SEQUENCES AREN'T OF LENGTH MAX_TIMESTEPS (Taken care of for now)

Check grad clipping and norm (Might do later)

Be able to pass attention and decoder to model (Not a good Idea)

Maybe remove all decoding functionality from model into seperate decoder (Not a good idea)

Reduce num duplicate params passed to decoderCell, model, and attention (Not necessary if everything is in a single model)

How to choose next word during validation? greedy search or beam search? (Done)

## Beam search stuff

According to tutorial, beam search while decoding

allennlp runs _forward_beam_search method when not training

_forward_beam_search runs allennlp's beam search code that takes in a take_step function at each beam search timestep

take_step is basicially embedding, my ImageCaptioningDecoder with a logit layer, and a log_softmax layer (why log_softmax?) (TODO: Combine embedding, decoder, and linear layer into one)

take_step takes in last predictions and state (dictionary of tensors) and returns class_log_probs and state

Modify code within the timestep to be run independently of timesteps loop

Run bleu score on beam searched predictions

LOSS IS ABOVE 0 EVEN IF PERFECT MATCH (Loss is calculated on logits, not argmaxed)

## Notes

will implement decoding on val like allennlp

Allennlp might not implement val loss correctly

validation max decoding steps are the same as length of target

beam search and bleu score is also implemented only on predictions with length of target_len

decode will run on model.forward_on_instances()

use predictor to run prediction and any post processing

preds returned from _decode will be sorted in training so use sorted labels and masks as well

first input to decoder at validation is start token

## Installation

### 1: Download dataset

### 2: Unzip dataset

`unzip processed-crohme-dataset.zip`  
`tar -xf 2013.tgz -C ./`

### 3: Make virtualenv

`virtualenv -p python3.6 environment`  
`source environment/bin/activate`

### 4: Create train-val splits

`python split_data.py`


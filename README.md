# math-expression-recognition

## Table Of Contents

* [Template](#template)
* [Todo](#todo)
* [Notes](#notes)
* [Transformers](#transformers)
* [Later](#later)
* [Done](#done)
* [Other](#other)
* [Beam Search](#beam-search-stuff)
* [Papers](#papers)
  * [Multi scale attention with dense encoder](#multi-scale-attention-with-dense-encoder)
  * [Image to markup generation with coarse to fine attention](#image-to-markup-generation-with-coarse-to-fine-attention)
* [Experiments](#experiments)
  * [Larger Decoder (Better)](#larger-decoder)
  * [Pretrained Resnet 18 (Better)](#pretrained-resnet18)
  * [20 epochs half lr after 10 (Better)](#20-epochs-half-lr-after-10)
  * [check doubly stochastic loss (Come back to later)](#check-doubly-stochastic-loss)
  * [normalizing image (Come back to later)](#normalizing-image)
  * [20 epochs half lr after 10 and doubly stochastic loss (Not better)](#20-epochs-half-lr-after-10-and-doubly-stochastic-loss)

## Template

  * [](#)
  
### 
Kernel:  
Results:

```

```

```

```

```

```

## ToDo

Check resizing/padding

Check grad clipping

Reduce lr on plateau (5? 3?)

Look into all math recognition papers

Increase the batch size

Larger resnet

Use im2latex dataset for pretraining http://lstm.seas.harvard.edu/latex/

## Notes

**Refer to fairseq**

Is label during beam search out of order? **No, all state vars are sorted in _decode() **

Tune beam size

will implement decoding on val like allennlp

Allennlp might not implement val loss correctly

validation max decoding steps are the same as length of target

beam search and bleu score is also implemented only on predictions with length of target_len

decode will run on model.forward_on_instances()

use predictor to run prediction and any post processing

preds returned from _decode will be sorted in training so use sorted labels and masks as well

first input to decoder at validation is start token

## Transformers

https://github.com/ruotianluo/Transformer_Captioning

https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html

> Paper: http://aclweb.org/anthology/P18-1238

https://openreview.net/forum?id=HJN6DiAcKQ

http://openaccess.thecvf.com/content_ICCV_2017/papers/Pedersoli_Areas_of_Attention_ICCV_2017_paper.pdf

https://www.researchgate.net/publication/325016817_Captioning_Transformer_with_Stacked_Attention_Modules

## Later

fp16 **Wait until allennlp supports apex**

Preprocess latex like https://github.com/harvardnlp/im2markup **Their preprocessing is for images of latex**

## Done

Check scaling image

Check doubly stochastic loss

Print out ground truth labels in predict

Pretrained resnet **Better**

Bigger model **Better**

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

## Papers

### Multi scale attention with dense encoder:

> Paper: https://arxiv.org/abs/1801.03530

> Github: https://github.com/JianshuZhang/WAP

> Details
* Train with lr of 1e-8 until metric doesn't improve

### Image to markup generation with coarse to fine attention:

> Paper: https://arxiv.org/pdf/1609.04938.pdf

> Github: https://github.com/harvardnlp/im2markup

> Details
 * Train starting with lr of 0.1 and halve when metric doesn't improve for a total of 12 epochs
 
## Experiments
 
### Larger decoder
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=10993888  
Result: Better

256 hidden units for all decoders

```
Metrics: {
  "best_epoch": 8,
  "peak_cpu_memory_MB": 3412.376,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "01:28:32",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_loss": 1.810625918276675,
  "training_cpu_memory_MB": 3412.376,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.05348376783490231,
  "validation_exprate": 0.0,
  "validation_loss": 1.8296998398644584,
  "best_validation_BLEU": 0.04266974779147556,
  "best_validation_exprate": 0.0,
  "best_validation_loss": 1.827181032725743
}
```

 ```
 {
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.1
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
 ```
 
### Pretrained Resnet18
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11019236  
Result: Better; Loss is lower

Pretrained resnet18 on imagenet

```
Metrics: {
  "best_epoch": 8,
  "peak_cpu_memory_MB": 3408.14,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "01:27:12",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_loss": 1.5743129854803686,
  "training_cpu_memory_MB": 3408.14,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.04362086240345169,
  "validation_exprate": 0.0,
  "validation_loss": 1.8315349945000239,
  "best_validation_BLEU": 0.060921997280835714,
  "best_validation_exprate": 0.0,
  "best_validation_loss": 1.6474940734250205
}
```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256,
        "pretrained": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.1
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
```

### 20 epochs half lr after 10
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11019311  
Results: A lot better; loss goes down!

```
Metrics: {
  "best_epoch": 19,
  "peak_cpu_memory_MB": 3424.544,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "02:49:55",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.8282530619217469,
  "training_cpu_memory_MB": 3424.544,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.08959688555673197,
  "validation_exprate": 0.0028296547821165816,
  "validation_loss": 1.1140256417649133,
  "best_validation_BLEU": 0.08959688555673197,
  "best_validation_exprate": 0.0028296547821165816,
  "best_validation_loss": 1.1140256417649133
}
```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256,
        "pretrained": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
```

### check doubly stochastic loss
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11019369  
Result: Not directly comparable since loss function has changed, but looks ok

```
Metrics: {
  "best_epoch": 9,
  "peak_cpu_memory_MB": 3395.48,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "01:26:24",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_loss": 1.683724222956477,
  "training_cpu_memory_MB": 3395.48,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.08684682558748881,
  "validation_exprate": 0.0005659309564233164,
  "validation_loss": 1.8228243717125483,
  "best_validation_BLEU": 0.08684682558748881,
  "best_validation_exprate": 0.0005659309564233164,
  "best_validation_loss": 1.8228243717125483
}
```

Code Changes:

```
state['loss'] += ((1 - torch.sum(state['attention_weights'], dim=1)) ** 2).mean()
```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256,
        "pretrained": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.1
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
```

### normalizing image
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config/data?scriptVersionId=11019369  
Results: Not much difference; Won't use for now, come back to it later if necessary

```
Metrics: {
  "best_epoch": 9,
  "peak_cpu_memory_MB": 3424.244,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "01:27:02",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_loss": 1.7649663300127596,
  "training_cpu_memory_MB": 3424.244,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.07426352194827664,
  "validation_exprate": 0.0,
  "validation_loss": 1.788192174264363,
  "best_validation_BLEU": 0.07426352194827664,
  "best_validation_exprate": 0.0,
  "best_validation_loss": 1.788192174264363
}
```

```
img = (img - self.mean) / self.std
```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256,
        "pretrained": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.1
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
```

### 20 epochs half lr after 10 and doubly stochastic loss
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11024682  
Results:

```
Metrics: {
  "best_epoch": 16,
  "peak_cpu_memory_MB": 3416.648,
  "peak_gpu_0_memory_MB": 10906,
  "training_duration": "02:48:36",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 1.4561049691191665,
  "training_cpu_memory_MB": 3416.648,
  "training_gpu_0_memory_MB": 10906,
  "validation_BLEU": 0.07566714071122081,
  "validation_exprate": 0.0028296547821165816,
  "validation_loss": 1.80103257724217,
  "best_validation_BLEU": 0.08142147376864764,
  "best_validation_exprate": 0.001697792869269949,
  "best_validation_loss": 1.790456669671195
}
```

```

```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20, 
        "embedding_dim": 256,
        "attention_dim": 256,
        "decoder_dim": 256,
        "pretrained": true
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 64
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
       "learning_rate_scheduler": {
            "type": "multi_step",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "directory_path": "/path/to/vocab"
#     },
}
```

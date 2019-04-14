# math-expression-recognition

## Table Of Contents

* [Template](#template)
* [Todo](#todo)
* [Done](#done)
* [Questions](#questions)
* [Notes](#notes)
* [Transformers](#transformers)
* [Later](#later)
* [Other](#other)
* [Beam Search](#beam-search-stuff)
* [Papers](#papers)
<details>
  <summary>Papers</summary>

  * [WAP](#wap)
  * [Multi scale attention with dense encoder](#multi-scale-attention-with-dense-encoder)
  * [Image to markup generation with coarse to fine attention](#image-to-markup-generation-with-coarse-to-fine-attention)
  * [Training an End-to-End System for Handwritten Mathematical Expression Recognition by Generated Patterns](#training-an-end-to-end-system-for-handwritten-mathematical-expression-recognition-by-generated-patterns)
  * [Image-to-Markup Generation via Paired Adversarial Learning](#image-to-markup-generation-via-paired-adversarial-learning)
  * [Image To Latex with DenseNet Encoder and Joint Attention](#image-to-latex-with-densenet-encoder-and-joint-attention)
  * [Robust Encoder-Decoder Learning Framework towards Offline Handwritten Mathematical Expression Recognition Based on Multi-Scale    Deep Neural Network](#robust-encoder-decoder-learning-framework-towards-offline-handwritten-mathematical-expression-recognition-based-on-multi-scale-deep-neural-network)
</details>
<details>
  <summary>Experiments</summary>

  <details>
    <summary>Original experiments</summary>
  
  * [Larger Decoder (Better)](#larger-decoder)
  * [Pretrained Resnet 18 (Better)](#pretrained-resnet18)
  * [20 epochs half lr after 10 (Better)](#20-epochs-half-lr-after-10)
  * [check doubly stochastic loss (Come back to later)](#check-doubly-stochastic-loss)
  * [normalizing image (Come back to later)](#normalizing-image)
  * [20 epochs half lr after 10 and doubly stochastic loss (Not better)](#20-epochs-half-lr-after-10-and-doubly-stochastic-loss)
  * [min count 10 (Better)](#min-count-10)
  * [512x256 (Slightly better)](#512x256)
  * [512x128 (No difference)](#512x128)
  * [reduce on plateau factor 0.5 patience 5 (Higher bleu score)](#reduce-on-plateau-factor-05-patience-5)
  * [resnet50 (Not better)](#resnet50)
  * [character tokenizer (Not better)](#character-tokenizer)
  * [no doubly stochastic attention (slightly worse)](#no-doubly-stochastic-attention)
  * [batch size 16 (Better)](#batch-size-16)
  * [lr 1e-3 (Not significatly better now)](#lr-1e-3)
  * [bleu as val metric (Better; but won't use just yet)](#bleu-as-val-metric)
  * [75 timesteps (Better)](#75-timesteps)
  * [Double everything (Doesn't help)](#double-everything)
  * [different img loading (Better; helps with model not learning anything)](#different-img-loading)
  * [different img loading cv2 (worse)](#different-img-loading-cv2)
  * [beam size 10)](#beam-size-10)
  * [fixed beam search (Fixes beam search; Better)](#fixed-beam-search)
  * [fix tokenization (Will use)](#fix-tokenization)
  * [smaller beam size and no min count (worse)](#smaller-beam-size-and-no-min-count)
  * [lr 0.1 patience 1 (worse)](#lr-01-patience-1)
  * [SGD momentum 0.9 (Better!!)](#sgd)
  * [min count](#min-count)
  * [doubly stochastic attention (better)](#doubly-stochastic-attention)
  * [30 epochs (better)](#30-epochs)
  * [50 epochs (better; best epoch 37)](#50-epochs)
  * [2x1 aspect ratio (Worse)](#2x1-aspect-ratio)
  * [encode image features (Better)](#encode-image-features)
  * [encode hidden state (Better)](#encode-hidden-state)
  * [encode image features and hidden state (Worse)](#encode-image-features-and-hidden-state)
  * [encode image features bidirectional (Worse)](#encode-image-features-bidirectional)
  * [fixed bidirectional image features (Better, but not good enough)](#fixed-bidirectional-image-features)
  * [3x3 inconv (Worse)](#3x3-inconv)
  * [no avg pooling (doesn't make a difference, Good)](#no-avg-pooling)
  </details>

  <details>
    <summary>Reaching baseline experiments</summary>
  
  * [gru row encoder (Worse)](#gru-row-encoder)
  * [gru row encoder 256 units (Better; but not good enough)](#gru-row-encoder-256-units)
  * [bidirectional row encoder (Worse)](#bidirectional-row-encoder)
  * [reversed bidirectional row encoder (Better than ^, but worse)](#reversed-bidirectional-row-encoder)
  * [vgg encoder (Failed)](#vgg-encoder)
  * [New baseline (1.775)](#new-baseline)
  * [lstm encoder (1.6892)](#lstm-encoder)
  * [im2latex encoder (1.73)](#im2latex-encoder)
  * [msa-decoder (1.77)](#msa-decoder)
  * [densenet encoder (2.22 but similar exprate)](#densenet-encoder)
  * [lstm and msa (1.69)](#lstm-and-msa)
  * [lstm encoder bidirectional no reverse (1.708)](#lstm-encoder-bidirectional-no-reverse)
  * [lstm encoder bidirectional reverse (1.699)](#lstm-encoder-bidirectional-reverse)
  * [im2latex encoder bidirectional no reverse (1.78)](#im2latex-encoder-bidirectional-no-reverse)
  * [lstm encoder 2 layers (1.70)](#lstm-encoder-2-layers)
  * [WAP backbone encoder (2.144)](#wap-backbone-encoder)
  * [Im2latex backbone enoder (2.40))](#im2latex-backbone-encoder)
  * [remove extra avg pool (1.636)](#remove-extra-avg-pool)
  * [small resnet 18 (2.12)](#small-resnet-18)
  * [downsample feature map (1.706)](#downsample-feature-map)
  * [not pretrained (1.835)](#not-pretrained)
  * [multi scale encoder (1.13, but similar to 1.636)](#multi-scale-encoder)
  * [multi scale lstm encoder (1.13; but slightly higher metrics)](#multi-scale-lstm-encoder)
  * [multiscale encoder and decoder (1.136; but slightly worse metrics)](#multiscale-encoder-and-decoder)

  * [**BEST** lstm lr 0.1](#lstm-lr-01)
  
  #### Redo architecture experiments with new lr and val metric exprate

  Baseline (v85; 0.31)

  Encoders:

  ##### LSTM encoder (v86; 0.3888)  
  Full vocab (v106; 0.357)  
  min token count 20 (v108; 0.3791; no difference)  
  256x1024 (v111; 0.3033)  
  BILSTM encoder (v112; 0.2869)  
  keep aspect ratio 512x512 (v125; 0.3157)  
  lstm no bias no teacher forcing (v8; 0.2642)  
  better tokenizer and full vocab (v9; 0.3834)  
  better tokenizer; no teacher forcing (v10; 0.2455)  
  batch size 8 (v12; 0.3880)  
  removing latex $ symbol (v13; 0.3903)  
  tanh in attention; nesterov; weight decay (v27; 0.3789)  
  no nesterov or weight decay and re-add doubly stochastic (v28; 0.3812)  
  SCSE (v38; 0.3981) **BETTER**  
  
  ##### Lstm Resnet 50
  256x1024 (v122; 0.2982)  
  256x1024 downsampled to 4x16 (v123; 0.3486)  
  128x512 (v124; 0.3684)  

  ##### Im2latex: **Stuck at 0.21; expected**  
  **Im2latex exact copy** (v92; 0.2173) *0.1 less since original is trained on external data*  
  Backbone and encoder (v87; 0.2286)  

  ##### WAP: **Stuck around 0.25**  
  WAP encoder (v88; 0.2196)  
  WAP exact copy (v95; 0.25)  
  Exact copy 1024x256 (v101; 0.21)  
  Full vocab (v104; 0.24)  
  correct convs and min_count 20 (v109; ~0.25)  
  grad clipping (v110; ~0.25)  
  lstm (v115; 0.2456; No change)  

  ##### Multiscale:  
  baseline (v90; 0.2530)  
  Exact copy except for dense encoder uses resnet 18 5x5 and 7x7 coverage (v126; 0.2801)  
  Exact copy (v133; 0.2092)  
  adam lr 1e-3 (v134; 0.1470)  
  adadelta lr 1e-8 (v135; 0)  
  lstm encoder (v116; 0.2863; Better, but not that much)  
  densenet msa no teacher forcing (v7; 0.052)  
  msa doesn't pass in previous timestep's predictions to gru; try with (v6; 0.242)  
  new tokenizer (v11; 0.1561)
  lr 0.01 30 epochs (v16; 0.07)
  dropout 0.2 (v22; 0.2048)  
  dropout + weight decay 1e-4 (v23; 0.2296)  
  dropout + weight decay + nesterov (v24; 0.2307)  
  tanh in attention (v26; 0.3065)  
  lstm + everything else (v30; 0.3416)  
  256x1024 batch size 2 (v31; takes ~3h/epoch)  
  Dropout after embedding (v36; 0.3484)  
  Scheduled sampling 0.5 (v35; 0.2748)  
  Scheduled sampling 0.8 (v37; 0.2895)  
  SCSE (v39; 0.3122)

  ##### Densenet
  densenet encoder (v91; 0.01)  

  ##### Small resnet18
  small resnet18 (v114; 0.2761; worse than baseline)  
  </details>

  <details>
    <summary>crohme 2019</summary>
  
  train on 2013 val on 2019 val(57; doesn't help)  
  train on 2013 val on 2019 train (58; Looks normal)  
  train on val val on train (60; graph is better; but loss doesn't go down enough (expected))  
  **problem most likely with small and out of distribution val data**  
  im2latex encoder (62; doesn't help by itself)  
  dropout 0.5 (v63; slightly better)  
  dropout + adam 1e-4 (v64; better; but val loss is still higher)  
  no teacher forcing (v65; doesn't help)  
  lr 1e-5 (v66; helps but too slow and probably stops after a few more epochs)  
  densenet backbone (v67; doesn't help)  
  resnet18 dropout (v70)  
  resnet18 not pretrained (v71; doesn't help)  
  embedding dropout (v72)  
  half decoder and attention hidden sizes to 128 (v73)  
  
  </details>

</details>


## Template

  * [](#)

###
Kernel:  
Results:

```
```
```
```

## ToDo

2019 data **NEW**

combine feature maps from all blocks

Pool feature maps and use somehow?

clr/sgdr ?

change lr scheduler from 0.5 to 0.1?

more regularization in resnet18

img augmentation

different attention mechanisms (add, concat, dot, etc)

Train encoder and decoder with different learning rates
* Related: https://github.com/allenai/allennlp/issues/2618

multiscale only gets 0.44 with a single model 

2013 is ~4% harder than 2014 https://github.com/JianshuZhang/WAP/issues/6#issuecomment-388676301

model is overfitting on train **Regularization helps**

transformer decoder

Larger/Smaller models

render predicted latex **Do later**

Use im2latex dataset for pretraining http://lstm.seas.harvard.edu/latex/ **Do later**

## Done

Get 2016 training data **Don't need**

Multiscale now learns more interesting attention maps but not perfect; main attention weights are in columns

Other regularization methods like kmnist + tgs **kmnist doesn't have anything**

check attention weights for softmax/scaling problems **This might be the big problem with current models** *Repeatedly can't find anything wrong with it*

scheduled sampling **Always lowers val score**

scse **Helps resnet more than densenet**

figure out why msa doesn't get higher exprate **Almost certainly confirmed to be because of overfitting**

Best model may be at max possible score? **Decreasing overfitting could increase score a bit**

more normalization to prevent overfitting **Helps a lot for msa, not as much for resnet**

missing tanh in attention **Tanh improves msa by a lot, also helps with attention visualizations**

params.pop **Only works when parent needs child's param; so can only use on nested encoders**

msa overfits more to train set **It's a larger model**

add regularization **weight decay, dropout, and nesterov increase msa by ~0.1**

weight decay **Helps ~0.02 on msa**

### Normalizing Ground truth: **Shouldn't make more than a ~1% difference**
* latex ground truth isn't normalized

* latex to pmml: https://github.com/JianshuZhang/WAP/issues/18#issuecomment-456792764

* use pandoc to convert latex to other formats to evaluate **Can't**

Why are attention maps uniform? (V18) **Attention attending to entire image**

no doubly stochastic attention (v19) **Attention attending to single part of image for all timesteps**

Check original attention implementation **correct**

run lr finder on msa **Done**

visualize raw attention heatmaps (v14) **No change between timesteps**

stop teacher forcing **no teacher forcing gets you ~0.25**

print overall accuracy x/x **Not necessary**

view predicted text and check if correct **It is**

some labels have $ symbol, others don't **removing all '$' helps increase exprate to 0.39**

use lstm to decrease feature map size (v4; 0.2409)

lstm 128x512 non teacher forcing (v2; 0.2567)

check attention **it's correct**

can't concat attention since encoder returns h\*w and decoder returns 1

lstm 256x512 (v1; 0.3687; https://www.kaggle.com/bkkaggle/math-recognition?scriptVersionId=12274376)

average size and aspect ratio (200x450; 2.4:1)

Use a gru to only need h_0? **Shouldn't be a huge difference**

vocab size all vs 10 vs 20 (437 vs 136 vs 127) (Not much difference)

batchnorm? **before or after relu or dropout** *bn -> relu -> dropout*

90:10 train val split best model (v129; 0.3902; slight increase)

gradient clipping **No difference**

vocab on train or val **Both by default**

check exprate function; try out official implementation? (No official implementation)

Try other people's code **Not worth it**

Round img values to nearest int (no difference; but use anyway)

correctly predicting h_0, c_0? **Yes**

Is masking padded data correct? **Yes again**

SGD lr 0.1 (1.4731; 16 epochs) **WAY BETTER**

Adam default params (1.509; 12 epochs)

remove relu from attention (1.641; v72)

im2latex style (2.11; v69); token embeddings 256 instead of 80 and different optimizer hyperparameters
exact im2latex copy except bidirectional (3.509; v70)
with bidirectional (6.177; v71)

check metrics functions **exprate looks right**

multiscale attention: (1.13)
 * doubly stochastic loss with multiscale **MSA doesn't use it; won't use either**
 * lstm encoder needs to be changed to work with two feature maps **1.13; but higher metrics**
 * msa encoder and decoder **1.13; but lower metrics**
 * msa dense encoder

make sure all subclasses use same params as superclass **Done**

rename base classes **Done**

rename all @register vars without -attention, etc **Done**

Not pretrained (1.835; worse)

Use avg pool to downsample feature map from (8,32) -> (4,16) (1.706; Big increase over below; try more epochs like below)

Remove last conv block from resnet18 (2.12; try more epochs (1.687))

Remove extra avg pool from resnet (1.636)

Im2latex backbone as encoder (Worse)

WAP backbone as encoder (Worse)

lstm encoder (Better):
* Bidirectional (reverse (Not better) or no reverse (Not better))
* layers (1, 2 (Not better))

im2latex encoder (Worse):
* Bidirectional (reverse (Won't try) or no reverse (Worse))

Densenet encoder (Higher loss)

VGG like small encoder (Worse)

more ifttt **Done**

View saved tensorboard logs **Will do when needed**

num params **Already shown in train logs**

Log attention visualizations to tensorboard **Done**

Log config to tensorboard **Won't work with old version of tensorboardX**

duplicate doubly stochastic attention in decoder **Done**

encode each row of feature map with rnn: **Worse**

 * bidirectional row encoder **Worse for me**
 * trainable initial hidden state for each row; "Positional embeddings" (Embedding with vocab size height of encoded image) **Better**
 * Add vs concat vs reverse + add **Add is better; concat isn't an option**
 * INFO on pytorch bidirectional add vs concat:
 * https://discuss.pytorch.org/t/concatenation-of-the-hidden-states-produced-by-a-bidirectional-lstm/3686/2
 * https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/5
 * Pretty sure im2latex either add or reverse + add bidirectional hidden states; add is more likely
 * Not sure what to pass as initial context state of lstm; paper only refers to hidden state **Use gru**

Check last few experiments to check if they were configured and evaluated correctly wrt other experiments **They are**

custom first conv with 3x3 kernel **Worse**

no avg pooling? **No change; good**

Reverse bidirectional image feature when adding **Better but not good enough**

encode image features with bilstm **Worse**

encode image features and hidden state **Worse**

encode image features with lstm **Better**

encode last timestep's hidden state with lstm **Better**

Rename image to math **Done**

resize images to 2:1 aspect ratio **Worse**

use doubly stochastic attention and loss *consider later; they alter attention heatmaps* **Done**

use values other than 0 for background? **No, shouldn't be necessary**

reshape img to right size

print out preds

Refactor prediction and visualization code

Show only right amount of timesteps

Show beam searched predictions at timesteps

Visualize attention heatmaps

Visualize source images of preds

use min count

cat metrics

no min count

use loss as validation metric

use smaller beam width **Worse**

split numbers

Lowercase **uppercase words might have special meaning**

`\\` vocab token

beam size **Better**

print top 50

Check grad clipping **Done**

Look into all math recognition papers

Check if the model actually learns anything? **It does; input img intensities had to be reversed**

input channels

Non pretrained **Won't do yet**

see raw train preds

More timesteps **Better**

Bigger model **Doesn't help yet**

Change the batch size **16 is better**

validaton metric **bleu is better, but won't use until model is working

train with lower lr **Slightly worse; more epochs?**

Doubly stochastic attention **Slightly worse**

Reduce lr on plateau (5? 3?)

maybe go back to character tokenization? **Not better**

fix pred printing

Larger resnet **Not better as of yet**

remove first conv? **No**

AvgPool rectangular

Adaptive avg pool right? **Yes**

Check resizing/padding/rectangular images **Slightly better**

Only use tokens that appear at least n times **Better**

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

## Questions

resnet vs vgg vs densenet

exprate at different architectures

image size dynamic

2016 vs 2013?

loss values

how to normalize

what techniques to reduce overfitting

scheduled sampling/teacher forcing

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

### WAP

> Paper: file:///C:/Users/Bilal/Documents/research/math%20expression%20recognition/Crohme-papers/WAP.pdf

> Github: https://github.com/JianshuZhang/WAP

> Details
 * 8-directional features
 * Coverage: conv over sum of past attention weights
 * exprate of 46.55 on 2014
 * exprate of 44.55 on 2016
 * exprate increases almost 0.1 with coverage
 * encoder output dim 128
 * attention dim 512
 * GRU decoder dim 256
 * coverage over past attention weights and attention over coverage
 * https://github.com/JianshuZhang/WAP/issues/2

### Multi scale attention with dense encoder:

> Paper: file:///C:/Users/Bilal/Documents/research/math%20expression%20recognition/Crohme-papers/Multi%20scale%20attention%20with%20dense%20encoder.pdf

> Github: https://github.com/JianshuZhang/WAP
> Reimplementation: https://github.com/whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition/blob/master/Attention_RNN.py
  * Gets 0.32 exprate

> Details
* Train with adadelta wd 1e-4 until metric doesn't improve
* Encode last hidden state with rnn before decoder cell
* exprate of ~0.5 on 2016

### Image to markup generation with coarse to fine attention:

> Paper: file:///C:/Users/Bilal/Documents/research/math%20expression%20recognition/Crohme-papers/Image%20to%20markup%20generation%20with%20coarse%20to%20fine%20attention.pdf

> Github: https://github.com/harvardnlp/im2markup
> Pytorch version: https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/im2text.md

> Details
 * Train starting with lr of 0.1 and halve when metric doesn't improve for a total of 12 epochs
 * Images are resized to ~200x50
 * Each row of the feature map is encoded with rnn then attention is computed
 * exprate on 2013: 33.53 (With external data) 
 * bidirectional row encoder
 * trainable initial hidden state for each row; "Positional embeddings" (Embedding with vocab size height of encoded image)
 * Pretty sure they either add or reverse + add bidirectional hidden states
 
**Im2latex exact copy** (v92; 0.2173) *0.1 less since original is trained on external data*  

 ### Training an End-to-End System for Handwritten Mathematical Expression Recognition by Generated Patterns
 
 > Paper: file:///C:/Users/Bilal/Documents/research/math%20expression%20recognition/Crohme-papers/ICDAR2017-(Training_an_End-to-End_System_for_Handwritten_Mathematical_expressions_by_generated_patterns).pdf
 
 > Details:
 * Encodes input features with a bilstm
 * Img augmentation
 * 2014: 35.19

 ### Image-to-Markup Generation via Paired Adversarial Learning
 
 > Paper: http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/376.pdf
 
 > Details:
 * Exprate on 2014: 0.39
 * Paired adversairal learning
 
 ### Image To Latex with DenseNet Encoder and Joint Attention
 
 > Paper: https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050919X00034/1-s2.0-S1877050919302686/main.pdf?x-amz-security-token=AgoJb3JpZ2luX2VjEGoaCXVzLWVhc3QtMSJGMEQCIHEypBtwzUARttTsICc6A5x9wMfWloxb%2B8ICsfliALw3AiA2vZbmY4bHTq%2FEA4PBkIoHqaboSejnORpZa1jZ6xwk%2ByraAwgzEAIaDDA1OTAwMzU0Njg2NSIMrDR1nz68CarUHDtUKrcDjjWl3uggyL1SW0T%2FeqptoOfJQF98ogu58%2F2XEf3EcTbi%2F38ArIAU%2FXWXDBwDhtCm78MvG1%2BQnG9zsSUIbydK2CA5IEjVNMvVnJ8CjhDotAXtOT4zNRU8OxFRSQXifCa8n7HYbfe%2BTBsB0ePJYYL%2FZsuRYLitY%2BB%2Bp9lv9JgVRLJR5Uo8GeLVKwF55%2FRNwnTDIKkO9zTvk4QogK5V2j1HOUP3JGo0QqKiVGUDrLzcgZs0AT4RI66j40qNUxCjn%2B8KGQheSI3jo0s57fg7IpLO2PGZAdyOh2H535so8yDptmVSdHoLHmlZjhppFOweeo7%2FY%2Biq7HBxKnfAP%2Bt1CKuBF%2FRGpgzXamSutmT5u0S2C4mzRxe0esOJzs%2BDKKR83Muu4byf8E1CPuhsRtz893kV%2BjO%2FlB0gRRJ%2F%2B9swte75D0tl9HBzPRPrLTwXEq%2FOnhrfPvJMKvvrbSoeVNXzfSv2Aji3piD2zWBsUgPKVO57cmz7sIYv0H%2FxMg%2B7JGFQIgRSC1JN68I6myFsNrz%2FIH4t9%2BlIUdwjEbwdVqHkj5zvm2%2FOmYvSjYwVWarStQOHIomyS5rAapGlwDCu7qjlBTq1Aa%2BSbviuIWDsvOrCWeoCmW6ENhMwdKY2w0zhDoQ3s7VBVJPfjqOxVHLMNxgmmp641LqmEnFYksQFabNAcIPCA1eAwgG67lWt60Eb2%2BrOILg7m%2F4A1yZIcODDcSqIzOpcSYFjZ8umYWAGnpJ76E6EKFk9em2%2FoV%2FIaeJlrmUPzK1loHSoqZEro5BzEj36nC%2BEwYv9RLevd0Vq8lyRFayPq7Z%2BV9J3HXiFkPoN8uJh6RUAPQSub28%3D&AWSAccessKeyId=ASIAQ3PHCVTYTAOSHDC5&Expires=1554661398&Signature=FKQMHcE49n5KPEOOTH8%2F%2BFRHF7w%3D&hash=a90e88a61acb8f8b690f1c5b29dcae4ceb57cdd76965707767aeb75d2a0055bc&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050919302686&tid=spdf-1ec0306d-0655-4015-a4c5-cb8d1e019a84&sid=84dca7bc6dca3446e248baa6913273052623gxrqa&type=client
 
> Details:

* im2latex dataset
* exprate 37%
* densenet encoder


### Robust Encoder-Decoder Learning Framework towards Offline Handwritten Mathematical Expression Recognition Based on Multi-Scale Deep Neural Network

> Paper: https://arxiv.org/ftp/arxiv/papers/1902/1902.05376.pdf

> Details:

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

### min count 10
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11033579  
Results: Better

```
Metrics: {
  "best_epoch": 12,
  "peak_cpu_memory_MB": 3420.536,
  "peak_gpu_0_memory_MB": 10899,
  "training_duration": "02:46:11",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5793307759203352,
  "training_cpu_memory_MB": 3420.536,
  "training_gpu_0_memory_MB": 10899,
  "validation_BLEU": 0.06647298746871495,
  "validation_exprate": 0.0050933786078098476,
  "validation_loss": 0.9654609880277089,
  "best_validation_BLEU": 0.06420508374862695,
  "best_validation_exprate": 0.006791171477079796,
  "best_validation_loss": 0.9295466244220734
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
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 512x256
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11034306  
Results: Better

```
Metrics: {
  "best_epoch": 12,
  "peak_cpu_memory_MB": 2652.968,
  "peak_gpu_0_memory_MB": 9890,
  "training_duration": "02:43:45",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.4642702749183586,
  "training_cpu_memory_MB": 2652.968,
  "training_gpu_0_memory_MB": 9890,
  "validation_BLEU": 0.06276211994669638,
  "validation_exprate": 0.001697792869269949,
  "validation_loss": 0.9946336512054715,
  "best_validation_BLEU": 0.06912969854662752,
  "best_validation_exprate": 0.001697792869269949,
  "best_validation_loss": 0.9179870252098355
```

```

```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 256,
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20,
        "encoder_height": 16,
        "encoder_width": 8,
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
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 512x128
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11057703  
Results: No difference, might be because the model still isn't learning the latex; try using wider aspect ratios after sure that the model works

```
Metrics: {
  "best_epoch": 13,
  "peak_cpu_memory_MB": 2415.388,
  "peak_gpu_0_memory_MB": 8644,
  "training_duration": "02:17:50",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5043626680030479,
  "training_cpu_memory_MB": 2415.388,
  "training_gpu_0_memory_MB": 8644,
  "validation_BLEU": 0.07481919181026701,
  "validation_exprate": 0.003395585738539898,
  "validation_loss": 1.0206764382975442,
  "best_validation_BLEU": 0.07505426950014889,
  "best_validation_exprate": 0.003961516694963215,
  "best_validation_loss": 0.927132785320282
}
```

```

```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "max_timesteps": 20,
        "encoder_height": 16,
        "encoder_width": 4,
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
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### Reduce on plateau factor 0.5 patience 5
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11060502  
Results: Better; Higher val bleu 

```
Metrics: {
  "best_epoch": 17,
  "peak_cpu_memory_MB": 2420.088,
  "peak_gpu_0_memory_MB": 8644,
  "training_duration": "05:50:41",
  "training_start_epoch": 0,
  "training_epochs": 49,
  "epoch": 49,
  "training_loss": 0.04867182831439349,
  "training_cpu_memory_MB": 2420.088,
  "training_gpu_0_memory_MB": 8644,
  "validation_BLEU": 0.1269965683674257,
  "validation_exprate": 0.0050933786078098476,
  "validation_loss": 1.316885103072439,
  "best_validation_BLEU": 0.12616028968765172,
  "best_validation_exprate": 0.005659309564233163,
  "best_validation_loss": 1.0380352096898215
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
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
        "num_epochs": 50,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
       "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### Resnet50
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11060551  
Results: Not better

```
Metrics: {
  "best_epoch": 15,
  "peak_cpu_memory_MB": 2531.904,
  "peak_gpu_0_memory_MB": 10425,
  "training_duration": "03:34:20",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5602430024662534,
  "training_cpu_memory_MB": 2531.904,
  "training_gpu_0_memory_MB": 10425,
  "validation_BLEU": 0.07483617344219004,
  "validation_exprate": 0.0011318619128466328,
  "validation_loss": 1.0261380757604326,
  "best_validation_BLEU": 0.06389102246869556,
  "best_validation_exprate": 0.003961516694963215,
  "best_validation_loss": 0.9615265663181033
}
```

```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet50',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
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
        "num_epochs": 20,
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
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### character tokenizer
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11065538  
Results: Not better

```
Metrics: {
  "best_epoch": 13,
  "peak_cpu_memory_MB": 2428.216,
  "peak_gpu_0_memory_MB": 8652,
  "training_duration": "02:17:47",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5576974188422298,
  "training_cpu_memory_MB": 2428.216,
  "training_gpu_0_memory_MB": 8652,
  "validation_BLEU": 0.10014560136932416,
  "validation_exprate": 0.0005659309564233164,
  "validation_loss": 0.9400272007499423,
  "best_validation_BLEU": 0.08305926681458821,
  "best_validation_exprate": 0.0028296547821165816,
  "best_validation_loss": 0.8667837998696736
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "character"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
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
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### no doubly stochastic attention
Kernel: V46 https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11083975  
Results: slightly worse

```
Metrics: {
  "best_epoch": 18,
  "peak_cpu_memory_MB": 2424.304,
  "peak_gpu_0_memory_MB": 8632,
  "training_duration": "02:12:22",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.48250934293678216,
  "training_cpu_memory_MB": 2424.304,
  "training_gpu_0_memory_MB": 8632,
  "validation_BLEU": 0.11427185889087516,
  "validation_exprate": 0.003395585738539898,
  "validation_loss": 1.08288340483393,
  "best_validation_BLEU": 0.12441254787132351,
  "best_validation_exprate": 0.0028296547821165816,
  "best_validation_loss": 1.0762296978916441
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 20,
        "embedding_dim": 256,
        "doubly_stochastic_attention": false,
        "attention_dim": 256,
        "decoder_dim": 256
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
            "lr": 0.001
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### batch size 16
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11086134 v49  
Results: Better; Loss is lower

```
2019-03-02 23:40:28,090 - INFO - allennlp.commands.evaluate - BLEU: 0.06955897954768596
2019-03-02 23:40:28,091 - INFO - allennlp.commands.evaluate - exprate: 0.0050933786078098476
2019-03-02 23:40:28,091 - INFO - allennlp.commands.evaluate - loss: 0.9873247804405453
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 20,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "learning_rate_scheduler": {
             "type": "reduce_on_plateau",
             "factor": 0.5,
             "patience": 5
#            "type": "multi_step",
#            "milestones": [10, 20, 30, 40],
#            "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lr 1e-3
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11086184 V50  
Results: Not significatly better now

```
Metrics: {
  "best_epoch": 15,
  "peak_cpu_memory_MB": 2426.076,
  "peak_gpu_0_memory_MB": 8644,
  "training_duration": "02:24:45",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.487113605211447,
  "training_cpu_memory_MB": 2426.076,
  "training_gpu_0_memory_MB": 8644,
  "validation_BLEU": 0.10456420856035056,
  "validation_exprate": 0.00792303338992643,
  "validation_loss": 1.120521530508995,
  "best_validation_BLEU": 0.12984879191769547,
  "best_validation_exprate": 0.0050933786078098476,
  "best_validation_loss": 1.074400497334344
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 20,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
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
            "lr": 0.001
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### bleu as val metric
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11087901 v51  
Results: Better; But use after model is working

```
Metrics: {
  "best_epoch": 14,
  "peak_cpu_memory_MB": 2390.612,
  "peak_gpu_0_memory_MB": 8644,
  "training_duration": "02:17:28",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5430574360731486,
  "training_cpu_memory_MB": 2390.612,
  "training_gpu_0_memory_MB": 8644,
  "validation_BLEU": 0.07345152212269282,
  "validation_exprate": 0.006791171477079796,
  "validation_loss": 0.9656095079013279,
  "best_validation_BLEU": 0.08023707957732544,
  "best_validation_exprate": 0.008488964346349746,
  "best_validation_loss": 0.9554811652217593
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 20,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
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
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 75 timesteps
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11091031 v53  
Results: Better; Higher bleu

```
Metrics: {
  "best_epoch": 19,
  "peak_cpu_memory_MB": 2155.596,
  "peak_gpu_0_memory_MB": 5398,
  "training_duration": "03:50:13",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 0.5110849919497158,
  "training_cpu_memory_MB": 2155.596,
  "training_gpu_0_memory_MB": 5398,
  "validation_BLEU": 0.14157844837004005,
  "validation_exprate": 0.009620826259196379,
  "validation_loss": 0.9945293479674572,
  "best_validation_BLEU": 0.14157844837004005,
  "best_validation_exprate": 0.009620826259196379,
  "best_validation_loss": 0.9945293479674572
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### Double everything
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11093072 v54  
Results:Doesn't help

```
2019-03-03 07:18:05,723 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-03 07:18:05,723 - INFO - allennlp.commands.evaluate - BLEU: 0.053972748235069194
2019-03-03 07:18:05,723 - INFO - allennlp.commands.evaluate - exprate: 0.003961516694963215
2019-03-03 07:18:05,723 - INFO - allennlp.commands.evaluate - loss: 1.3646348623542097
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 1024,
        "width": 256,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 32,
        "encoder_width": 8,
        "max_timesteps": 75,
        "embedding_dim": 512,
        "doubly_stochastic_attention": true,
        "attention_dim": 512,
        "decoder_dim": 512
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### different img loading
Kernel:https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11114322 v55  
Results: Better;

```
2019-03-03 19:46:57,262 - INFO - allennlp.commands.evaluate - BLEU: 0.18469591785909167
2019-03-03 19:46:57,262 - INFO - allennlp.commands.evaluate - exprate: 0.04640633842671194
2019-03-03 19:46:57,262 - INFO - allennlp.commands.evaluate - loss: 0.9052463579285253
```
```
        img = (1 - plt.imread(path)[:,:,0])
        img = img.reshape(1, img.shape[0], img.shape[1])
        img = np.concatenate((img, img, img))
        img = cv2.resize(img.transpose(1, 2, 0), (self.height, self.width)).transpose(2, 0, 1)

```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### different img loading cv2
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11116779 v56  
Results: Worse

```
2019-03-03 23:05:47,797 - INFO - allennlp.commands.evaluate - BLEU: 0.12269647588672848
2019-03-03 23:05:47,797 - INFO - allennlp.commands.evaluate - exprate: 0.013016411997736276
2019-03-03 23:05:47,797 - INFO - allennlp.commands.evaluate - loss: 0.9937572060404597
```
```
        img = cv2.imread(path)
        img = img / 255
        img = 1 - img
        img = cv2.resize(img, (self.height, self.width))
        img = img.reshape(3, self.height, self.width)
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### beam size 10
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11124199 v58  
Results: Better, but really doesn't mean anything since beam search is fixed in the next experiment

```
2019-03-04 04:09:25,679 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-04 04:09:25,679 - INFO - allennlp.commands.evaluate - BLEU: 0.128425063311222
2019-03-04 04:09:25,679 - INFO - allennlp.commands.evaluate - exprate: 0.04640633842671194
2019-03-04 04:09:25,679 - INFO - allennlp.commands.evaluate - loss: 0.9284863885458525
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### fixed beam search
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11125346 v59  
Results:

```
2019-03-04 04:59:23,734 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-04 04:59:23,734 - INFO - allennlp.commands.evaluate - BLEU: 0.22114275113523937
2019-03-04 04:59:23,734 - INFO - allennlp.commands.evaluate - exprate: 0.11544991511035653
2019-03-04 04:59:23,734 - INFO - allennlp.commands.evaluate - loss: 0.8966405372898858
```
```
            state['h'], state['c'] = self._init_hidden(state['x'])
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```
### fix tokenization
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11158430 v60  
Results: Higher loss and lower bleu score but correct tokenization

```
2019-03-05 01:21:16,778 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-05 01:21:16,778 - INFO - allennlp.commands.evaluate - BLEU: 0.18163607123755118
2019-03-05 01:21:16,778 - INFO - allennlp.commands.evaluate - exprate: 0.1590265987549519
2019-03-05 01:21:16,778 - INFO - allennlp.commands.evaluate - loss: 1.1946798111941364
```
```
    def _tokenize(self, text):
        
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('{', ' { ')
        text = text.replace('}', ' } ')
        text = text.replace('$', ' $ ')
        text = text.replace('_', ' _ ')
        text = text.replace('^', ' ^ ')
        text = text.replace('+', ' + ')
        text = text.replace('-', ' - ')
        text = text.replace('/', ' / ')
        text = text.replace('*', ' * ')
        text = text.replace('=', ' = ')
        text = text.replace('[', ' [ ')
        text = text.replace(']', ' ] ')
        text = text.replace('|', ' | ')
        text = text.replace('!', ' ! ')
        text = text.replace(',', ' , ')
        
        text = text.replace('\\', ' \\')
        
        text = text.replace('0', ' 0 ')
        text = text.replace('1', ' 1 ')
        text = text.replace('2', ' 2 ')
        text = text.replace('3', ' 3 ')
        text = text.replace('4', ' 4 ')
        text = text.replace('5', ' 5 ')
        text = text.replace('6', ' 6 ')
        text = text.replace('7', ' 7 ')
        text = text.replace('8', ' 8 ')
        text = text.replace('9', ' 9 ')

        return [Token(token) for token in text.split()]

```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
        "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### smaller beam size and no min count
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11162145 v61  
Results: Worse

```
2019-03-05 04:08:07,499 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-05 04:08:07,499 - INFO - allennlp.commands.evaluate - BLEU: 0.052254567852234896
2019-03-05 04:08:07,499 - INFO - allennlp.commands.evaluate - exprate: 0.01867572156196944
2019-03-05 04:08:07,499 - INFO - allennlp.commands.evaluate - loss: 1.4841137861346339
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 3,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.01
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "min_count": {
#             'tokens': 10
#         }
#         "directory_path": "/path/to/vocab"
#     },
}
```

### lr 0.1 patience 1
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11185674 v62  
Results:

```
2019-03-05 17:08:11,030 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-05 17:08:11,030 - INFO - allennlp.commands.evaluate - BLEU: 8.993769912683115e-08
2019-03-05 17:08:11,030 - INFO - allennlp.commands.evaluate - exprate: 0.0
2019-03-05 17:08:11,030 - INFO - allennlp.commands.evaluate - loss: 4.900734297864072
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.1
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 1
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "min_count": {
#             'tokens': 10
#         }
#         "directory_path": "/path/to/vocab"
#     },
}
```
### sgd
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11185748 v63  
Results:

```
2019-03-05 17:11:14,813 - INFO - allennlp.commands.evaluate - Metrics:
2019-03-05 17:11:14,813 - INFO - allennlp.commands.evaluate - BLEU: 0.420855988359163
2019-03-05 17:11:14,813 - INFO - allennlp.commands.evaluate - exprate: 0.1007357102433503
2019-03-05 17:11:14,813 - INFO - allennlp.commands.evaluate - loss: 1.3468512232239183
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
#     "vocabulary": {
#         "min_count": {
#             'tokens': 10
#         }
#         "directory_path": "/path/to/vocab"
#     },
}
```

### min count
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11194990 v64  
Results:

```
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 512,
        "width": 128,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### doubly stochastic attention
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11329944 v80  
Results: Better

```
{
  "best_epoch": 19,
  "peak_cpu_memory_MB": 2695.14,
  "peak_gpu_0_memory_MB": 1481,
  "training_duration": "01:26:48",
  "training_start_epoch": 0,
  "training_epochs": 19,
  "epoch": 19,
  "training_loss": 1.105314669431065,
  "training_cpu_memory_MB": 2695.14,
  "training_gpu_0_memory_MB": 1481,
  "validation_BLEU": 0.4384247738592014,
  "validation_exprate": 0.11941143180531975,
  "validation_loss": 1.7994500860437617,
  "best_validation_BLEU": 0.4384247738592014,
  "best_validation_exprate": 0.11941143180531975,
  "best_validation_loss": 1.7994500860437617
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 20,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 30 epochs
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11330549 v81  
Results: Better;

```
{
  "best_epoch": 27,
  "peak_cpu_memory_MB": 2695.224,
  "peak_gpu_0_memory_MB": 1481,
  "training_duration": "02:09:20",
  "training_start_epoch": 0,
  "training_epochs": 29,
  "epoch": 29,
  "training_loss": 0.8273943196055037,
  "training_cpu_memory_MB": 2695.224,
  "training_gpu_0_memory_MB": 1481,
  "validation_BLEU": 0.48553515163211886,
  "validation_exprate": 0.16921335597057158,
  "validation_loss": 1.7258416113552746,
  "best_validation_BLEU": 0.4841102772382667,
  "best_validation_exprate": 0.16355404640633842,
  "best_validation_loss": 1.7124873668223888
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 30,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 6,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 50 epochs
Kernel:  v83  
Results: Better

```
{
  "best_epoch": 37,
  "peak_cpu_memory_MB": 2698.308,
  "peak_gpu_0_memory_MB": 1481,
  "training_duration": "03:26:32",
  "training_start_epoch": 0,
  "training_epochs": 49,
  "epoch": 49,
  "training_loss": 0.619864522399406,
  "training_cpu_memory_MB": 2698.308,
  "training_gpu_0_memory_MB": 1481,
  "validation_BLEU": 0.5237166993736423,
  "validation_exprate": 0.2037351443123939,
  "validation_loss": 1.7937162706443854,
  "best_validation_BLEU": 0.536373875131854,
  "best_validation_exprate": 0.21675155631013016,
  "best_validation_loss": 1.704390459232502
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 4,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 50,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
#             "type": "multi_step",
#             "milestones": [10, 20, 30, 40],
#             "gamma": 0.5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 2x1 aspect ratio
Kernel: https://www.kaggle.com/bkkaggle/allennlp-config?scriptVersionId=11374251 v84  
Results: Worse

```
{
  "best_epoch": 31,
  "peak_cpu_memory_MB": 2785.936,
  "peak_gpu_0_memory_MB": 2159,
  "training_duration": "03:53:37",
  "training_start_epoch": 0,
  "training_epochs": 49,
  "epoch": 49,
  "training_loss": 0.8311249016366933,
  "training_cpu_memory_MB": 2785.936,
  "training_gpu_0_memory_MB": 2159,
  "validation_BLEU": 0.4629746433532945,
  "validation_exprate": 0.15166949632144877,
  "validation_loss": 2.116068430849024,
  "best_validation_BLEU": 0.4300333402055585,
  "best_validation_exprate": 0.13978494623655913,
  "best_validation_loss": 2.0138201047708324
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 256,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "math-image-captioning",
        "encoder_type": 'resnet18',
        "pretrained": true,
        "encoder_height": 16,
        "encoder_width": 8,
        "max_timesteps": 75,
        "beam_size": 10,
        "embedding_dim": 256,
        "doubly_stochastic_attention": true,
        "attention_dim": 256,
        "decoder_dim": 256
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 50,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### encode image features
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11461311 v1  
Results: Better

```
{
  "best_epoch": 32,
  "peak_cpu_memory_MB": 2708.044,
  "peak_gpu_0_memory_MB": 1501,
  "training_duration": "02:58:50",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7506400365635281,
  "training_cpu_memory_MB": 2708.044,
  "training_gpu_0_memory_MB": 1501,
  "validation_BLEU": 0.552877996250616,
  "validation_exprate": 0.2693831352574986,
  "validation_loss": 1.6795133416717116,
  "best_validation_BLEU": 0.5445483999399532,
  "best_validation_exprate": 0.2591963780418789,
  "best_validation_loss": 1.621857388599499
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'lstm',
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true
            },
            "hidden_size": 512,
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### encode hidden state
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11489654 v5  
Results: Better

```
{
  "best_epoch": 35,
  "peak_cpu_memory_MB": 2726.884,
  "peak_gpu_0_memory_MB": 1517,
  "training_duration": "03:07:06",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.6874275887174304,
  "training_cpu_memory_MB": 2726.884,
  "training_gpu_0_memory_MB": 1517,
  "validation_BLEU": 0.5846165226826109,
  "validation_exprate": 0.27504244482173174,
  "validation_loss": 1.6777625062444188,
  "best_validation_BLEU": 0.5821900393791162,
  "best_validation_exprate": 0.2722127900396152,
  "best_validation_loss": 1.6422494188085333
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true                
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### encode image features and hidden state
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11489754 v6  
Results: Worse

```
{
  "best_epoch": 34,
  "peak_cpu_memory_MB": 2715.056,
  "peak_gpu_0_memory_MB": 1503,
  "training_duration": "03:02:29",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.8380908595221075,
  "training_cpu_memory_MB": 2715.056,
  "training_gpu_0_memory_MB": 1503,
  "validation_BLEU": 0.5324456424223838,
  "validation_exprate": 0.23486134691567628,
  "validation_loss": 1.7174942772667687,
  "best_validation_BLEU": 0.5224509928136901,
  "best_validation_exprate": 0.23429541595925296,
  "best_validation_loss": 1.6811244401845846
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true                
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "msa-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### encode image features bidirectional
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11489940 v7   
Results: Worse

```
{
  "best_epoch": 33,
  "peak_cpu_memory_MB": 2725.508,
  "peak_gpu_0_memory_MB": 1517,
  "training_duration": "03:03:15",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.6923653325613808,
  "training_cpu_memory_MB": 2725.508,
  "training_gpu_0_memory_MB": 1517,
  "validation_BLEU": 0.5725510916952119,
  "validation_exprate": 0.2693831352574986,
  "validation_loss": 1.7210015735110722,
  "best_validation_BLEU": 0.5621851301574926,
  "best_validation_exprate": 0.26089417091114886,
  "best_validation_loss": 1.6640424964664218
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true                
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### fixed bidirectional image features
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11511988 v8  
Results: Better than previous, but worse than best

```
{
  "best_epoch": 39,
  "peak_cpu_memory_MB": 2725.284,
  "peak_gpu_0_memory_MB": 1517,
  "training_duration": "03:07:54",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7793162158981168,
  "training_cpu_memory_MB": 2725.284,
  "training_gpu_0_memory_MB": 1517,
  "validation_BLEU": 0.5702787243312304,
  "validation_exprate": 0.26825127334465193,
  "validation_loss": 1.668770626858548,
  "best_validation_BLEU": 0.5702787243312304,
  "best_validation_exprate": 0.26825127334465193,
  "best_validation_loss": 1.668770626858548
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true                
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### 3x3 inconv
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11553644 v10  
Results: Worse

```
{
  "best_epoch": 29,
  "peak_cpu_memory_MB": 2730.412,
  "peak_gpu_0_memory_MB": 3497,
  "training_duration": "04:02:08",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.1236782585064211,
  "training_cpu_memory_MB": 2730.412,
  "training_gpu_0_memory_MB": 3497,
  "validation_BLEU": 0.47243654416987246,
  "validation_exprate": 0.20316921335597057,
  "validation_loss": 2.20243712803265,
  "best_validation_BLEU": 0.43021217181796484,
  "best_validation_exprate": 0.16864742501414828,
  "best_validation_loss": 2.1513477134275005
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 8,
                "encoder_width": 32,
                "pretrained": true,
                "custom_in_conv": true
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### no avg pooling
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11576248 v13  
Results: No change; Good

```
{
  "best_epoch": 35,
  "peak_cpu_memory_MB": 2723.636,
  "peak_gpu_0_memory_MB": 1517,
  "training_duration": "03:02:46",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7321721876368803,
  "training_cpu_memory_MB": 2723.636,
  "training_gpu_0_memory_MB": 1517,
  "validation_BLEU": 0.5696917663314149,
  "validation_exprate": 0.2705149971703452,
  "validation_loss": 1.6747943077001486,
  "best_validation_BLEU": 0.5584485780871594,
  "best_validation_exprate": 0.2614601018675722,
  "best_validation_loss": 1.6364673859364278
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### gru row encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11621828 v17  
Results: Worse

```
{
  "best_epoch": 32,
  "peak_cpu_memory_MB": 2707.028,
  "peak_gpu_0_memory_MB": 1495,
  "training_duration": "02:58:16",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.6961522034929888,
  "training_cpu_memory_MB": 2707.028,
  "training_gpu_0_memory_MB": 1495,
  "validation_BLEU": 0.5153442739581302,
  "validation_exprate": 0.20430107526881722,
  "validation_loss": 1.7591714075019769,
  "best_validation_BLEU": 0.4971638464552332,
  "best_validation_exprate": 0.20090548953027731,
  "best_validation_loss": 1.6868121366243105
}
```
```
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512,
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### gru row encoder 256 units
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments/data?scriptVersionId=11631017 v19  
Results: Better than previous; still not better than baseline

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2706.208,
  "peak_gpu_0_memory_MB": 1483,
  "training_duration": "02:48:12",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7056815314077144,
  "training_cpu_memory_MB": 2706.208,
  "training_gpu_0_memory_MB": 1483,
  "validation_BLEU": 0.5106420258942483,
  "validation_exprate": 0.2144878324844369,
  "validation_loss": 1.756954265070391,
  "best_validation_BLEU": 0.5121589619488466,
  "best_validation_exprate": 0.2144878324844369,
  "best_validation_loss": 1.7092307269036233
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 256,
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 256,#512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```
  
### bidirectional row encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11635961 v21  
Results: Worse

```
{
  "best_epoch": 26,
  "peak_cpu_memory_MB": 2703.776,
  "peak_gpu_0_memory_MB": 1489,
  "training_duration": "02:51:11",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.6700914836577161,
  "training_cpu_memory_MB": 2703.776,
  "training_gpu_0_memory_MB": 1489,
  "validation_BLEU": 0.5088521130107576,
  "validation_exprate": 0.18902093944538767,
  "validation_loss": 1.8061693283888671,
  "best_validation_BLEU": 0.45641710998488655,
  "best_validation_exprate": 0.15449915110356535,
  "best_validation_loss": 1.7392937342325847
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 256,
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 256,#512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### reversed bidirectional row encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11639778 v22  
Results: Better than non reversed bidirectional, but worse than non bidirectional

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2702.792,
  "peak_gpu_0_memory_MB": 1487,
  "training_duration": "02:43:01",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7076727373028233,
  "training_cpu_memory_MB": 2702.792,
  "training_gpu_0_memory_MB": 1487,
  "validation_BLEU": 0.48874979640617267,
  "validation_exprate": 0.19128466327108093,
  "validation_loss": 1.7938354112006523,
  "best_validation_BLEU": 0.4958804156988936,
  "best_validation_exprate": 0.19807583474816073,
  "best_validation_loss": 1.7431748010016777
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'resnet',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 256,
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 256,#512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256,
            "doubly_stochastic_attention": true
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### vgg encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11714309 v26  
Results: Failed

```
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'vgg',
            "encoder_type": 'vgg16',
            "encoder_height": 4,
            "encoder_width": 16,
            "pretrained": true,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "msa-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### new baseline
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11723206 v28  
Results: OK

```
{
  "best_epoch": 38,
  "peak_cpu_memory_MB": 2705.504,
  "peak_gpu_0_memory_MB": 1479,
  "training_duration": "02:35:33",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.0644252321838794,
  "training_cpu_memory_MB": 2705.504,
  "training_gpu_0_memory_MB": 1479,
  "validation_BLEU": 0.4346189755582612,
  "validation_exprate": 0.16411997736276174,
  "validation_loss": 1.795390358916274,
  "best_validation_BLEU": 0.43352269478593225,
  "best_validation_exprate": 0.16468590831918506,
  "best_validation_loss": 1.7752251764675517
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'backbone',
            "encoder_type": 'resnet18',
            "encoder_height": 4,
            "encoder_width": 16,
            "pretrained": true,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11728189 v35  
Results: Better

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2710.908,
  "peak_gpu_0_memory_MB": 1499,
  "training_duration": "02:51:44",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.1359341515405146,
  "training_cpu_memory_MB": 2710.908,
  "training_gpu_0_memory_MB": 1499,
  "validation_BLEU": 0.4251209276848993,
  "validation_exprate": 0.16921335597057158,
  "validation_loss": 1.7454834199166513,
  "best_validation_BLEU": 0.4452744087548317,
  "best_validation_exprate": 0.2014714204867006,
  "best_validation_loss": 1.6892793468526892
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### im2latex encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11751138 v36  
Results: Worse

```
{
  "best_epoch": 38,
  "peak_cpu_memory_MB": 2710.012,
  "peak_gpu_0_memory_MB": 1491,
  "training_duration": "02:52:14",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.1476542361721194,
  "training_cpu_memory_MB": 2710.012,
  "training_gpu_0_memory_MB": 1491,
  "validation_BLEU": 0.4305185428847582,
  "validation_exprate": 0.200339558573854,
  "validation_loss": 1.7398730054631963,
  "best_validation_BLEU": 0.4186060126765994,
  "best_validation_exprate": 0.19864176570458403,
  "best_validation_loss": 1.735885628708848
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### msa decoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11761029 v37  
Results: Not as good, why?

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2690.792,
  "peak_gpu_0_memory_MB": 1489,
  "training_duration": "02:48:23",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.0838393571689657,
  "training_cpu_memory_MB": 2690.792,
  "training_gpu_0_memory_MB": 1489,
  "validation_BLEU": 0.4205424116071523,
  "validation_exprate": 0.16355404640633842,
  "validation_loss": 1.8273741769361067,
  "best_validation_BLEU": 0.40925322499626376,
  "best_validation_exprate": 0.1658177702320317,
  "best_validation_loss": 1.7702094219826363
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'backbone',
            "encoder_type": 'resnet18',
            "encoder_height": 4,
            "encoder_width": 16,
            "pretrained": true,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "msa-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 10,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### densenet encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11767602 v39  
Results: higher loss, but similar bleu and exprate

```
{
  "best_epoch": 30,
  "peak_cpu_memory_MB": 2695.172,
  "peak_gpu_0_memory_MB": 16017,
  "training_duration": "05:46:15",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.128351241485026,
  "training_cpu_memory_MB": 2695.172,
  "training_gpu_0_memory_MB": 16017,
  "validation_BLEU": 0.4100378826948628,
  "validation_exprate": 0.15280135823429541,
  "validation_loss": 2.312613185461577,
  "best_validation_BLEU": 0.4090073179044538,
  "best_validation_exprate": 0.1420486700622524,
  "best_validation_loss": 2.222365846505036
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'backbone',
            "encoder_type": 'densenetMSA',
            "encoder_height": 8,
            "encoder_width": 32,
            "pretrained": false,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 1356, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm and msa
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11770601 v41  
Results: Worse

```
{
  "best_epoch": 39,
  "peak_cpu_memory_MB": 2699.896,
  "peak_gpu_0_memory_MB": 1501,
  "training_duration": "01:43:59",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.1725350708713358,
  "training_cpu_memory_MB": 2699.896,
  "training_gpu_0_memory_MB": 1501,
  "validation_BLEU": 0.4056459938803575,
  "validation_exprate": 0.19241652518392757,
  "validation_loss": 1.6966624399563213,
  "best_validation_BLEU": 0.4056459938803575,
  "best_validation_exprate": 0.19241652518392757,
  "best_validation_loss": 1.6966624399563213
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "msa-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm encoder bidirectional no reverse
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments/data?scriptVersionId=11793481 v43  
Results: Worse loss; little higher metrics

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2720.008,
  "peak_gpu_0_memory_MB": 1515,
  "training_duration": "01:48:43",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.0920721149552461,
  "training_cpu_memory_MB": 2720.008,
  "training_gpu_0_memory_MB": 1515,
  "validation_BLEU": 0.4642388256643965,
  "validation_exprate": 0.20543293718166383,
  "validation_loss": 1.747305524241817,
  "best_validation_BLEU": 0.4652358379891482,
  "best_validation_exprate": 0.2082625919637804,
  "best_validation_loss": 1.7082463223654945
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm encoder bidirectional reverse
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11809823 v44  
Results: Not better

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2719.652,
  "peak_gpu_0_memory_MB": 1515,
  "training_duration": "02:10:22",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.1355797504138083,
  "training_cpu_memory_MB": 2719.652,
  "training_gpu_0_memory_MB": 1515,
  "validation_BLEU": 0.448015105532121,
  "validation_exprate": 0.20656479909451048,
  "validation_loss": 1.7041660343204532,
  "best_validation_BLEU": 0.43883514544612573,
  "best_validation_exprate": 0.2071307300509338,
  "best_validation_loss": 1.6998484370944735
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### im2latex encoder bidirectional no reverse
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11809941 v45  
Results: Not better

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2722.064,
  "peak_gpu_0_memory_MB": 1543,
  "training_duration": "02:00:54",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.146014416784183,
  "training_cpu_memory_MB": 2722.064,
  "training_gpu_0_memory_MB": 1543,
  "validation_BLEU": 0.42446887790815874,
  "validation_exprate": 0.18166383701188454,
  "validation_loss": 1.8029189743437208,
  "best_validation_BLEU": 0.4452330041894771,
  "best_validation_exprate": 0.17713638936049803,
  "best_validation_loss": 1.783592737472809
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "im2latex",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm encoder 2 layers
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11809994 v46  
Results: Not better

```
{
  "best_epoch": 38,
  "peak_cpu_memory_MB": 2776.204,
  "peak_gpu_0_memory_MB": 1671,
  "training_duration": "01:58:51",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.2119314309159015,
  "training_cpu_memory_MB": 2776.204,
  "training_gpu_0_memory_MB": 1671,
  "validation_BLEU": 0.4021780161673791,
  "validation_exprate": 0.189586870401811,
  "validation_loss": 1.7044743643150673,
  "best_validation_BLEU": 0.4046472627041309,
  "best_validation_exprate": 0.18902093944538767,
  "best_validation_loss": 1.7007347313133445
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 2,
            "bidirectional": true
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### WAP backbone encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11810822 v47  
Results: 2.14

```
{
  "best_epoch": 37,
  "peak_cpu_memory_MB": 2574.18,
  "peak_gpu_0_memory_MB": 3839,
  "training_duration": "02:07:52",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.5559201550699466,
  "training_cpu_memory_MB": 2574.18,
  "training_gpu_0_memory_MB": 3839,
  "validation_BLEU": 0.3135305133877175,
  "validation_exprate": 0.13695529145444255,
  "validation_loss": 2.147862047762484,
  "best_validation_BLEU": 0.2900647413818954,
  "best_validation_exprate": 0.1245048104131296,
  "best_validation_loss": 2.144687180046563
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'backbone',
            "encoder_type": 'WAP',
            "encoder_height": 8,
            "encoder_width": 32,
            "pretrained": true,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 128, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### im2latex backbone encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11812125 v48  
Results: Not better

```
{
  "best_epoch": 31,
  "peak_cpu_memory_MB": 2630.544,
  "peak_gpu_0_memory_MB": 4147,
  "training_duration": "02:59:57",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.6092503421446855,
  "training_cpu_memory_MB": 2630.544,
  "training_gpu_0_memory_MB": 4147,
  "validation_BLEU": 0.23920498060894252,
  "validation_exprate": 0.06734578381437464,
  "validation_loss": 2.458981562305141,
  "best_validation_BLEU": 0.20928067807681497,
  "best_validation_exprate": 0.05263157894736842,
  "best_validation_loss": 2.406644803983671
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": 'backbone',
            "encoder_type": 'Im2latex',
            "encoder_height": 14,
            "encoder_width": 62,
            "pretrained": false,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### remove extra avg pool
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11856132 v53  
Results: Better

```
{
  "best_epoch": 33,
  "peak_cpu_memory_MB": 2705.844,
  "peak_gpu_0_memory_MB": 1499,
  "training_duration": "01:41:30",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.7535412389768195,
  "training_cpu_memory_MB": 2705.844,
  "training_gpu_0_memory_MB": 1499,
  "validation_BLEU": 0.5518747903715289,
  "validation_exprate": 0.2654216185625354,
  "validation_loss": 1.6942083105310664,
  "best_validation_BLEU": 0.5618913934686638,
  "best_validation_exprate": 0.2705149971703452,
  "best_validation_loss": 1.6360182171469335
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### small resnet 18
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11856466 v54  
Results: 2.12; probably needs more epochs

```
{
  "best_epoch": 36,
  "peak_cpu_memory_MB": 2692.096,
  "peak_gpu_0_memory_MB": 1525,
  "training_duration": "01:51:03",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.6009607727711017,
  "training_cpu_memory_MB": 2692.096,
  "training_gpu_0_memory_MB": 1525,
  "validation_BLEU": 0.33237756320058653,
  "validation_exprate": 0.15053763440860216,
  "validation_loss": 2.135711184493056,
  "best_validation_BLEU": 0.2978331278712403,
  "best_validation_exprate": 0.13582342954159593,
  "best_validation_loss": 2.1211826425414904
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'smallResnet18',
                "encoder_height": 8,
                "encoder_width": 32,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 256, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 256, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### downsample feature map
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11856676 v55  
Results: 1.706

```
{
  "best_epoch": 39,
  "peak_cpu_memory_MB": 2701.564,
  "peak_gpu_0_memory_MB": 1437,
  "training_duration": "01:43:16",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 1.2705452269558453,
  "training_cpu_memory_MB": 2701.564,
  "training_gpu_0_memory_MB": 1437,
  "validation_BLEU": 0.37160467788392215,
  "validation_exprate": 0.17487266553480477,
  "validation_loss": 1.7060529090262748,
  "best_validation_BLEU": 0.37160467788392215,
  "best_validation_exprate": 0.17487266553480477,
  "best_validation_loss": 1.7060529090262748
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'smallResnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 256, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 256, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### not pretrained
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11856749 v56  
Results: 1.835

```
{
  "best_epoch": 33,
  "peak_cpu_memory_MB": 2704.68,
  "peak_gpu_0_memory_MB": 1499,
  "training_duration": "01:41:59",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.8682422310248759,
  "training_cpu_memory_MB": 2704.68,
  "training_gpu_0_memory_MB": 1499,
  "validation_BLEU": 0.44884223052834626,
  "validation_exprate": 0.17826825127334464,
  "validation_loss": 1.8906518274599367,
  "best_validation_BLEU": 0.4484330879487077,
  "best_validation_exprate": 0.18053197509903793,
  "best_validation_loss": 1.8356320525074865
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": false,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'image-captioning-attention',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### multi scale encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11907049 v64  
Results: 1.13; but loss values aren't reliable

```
{
  "best_epoch": 29,
  "peak_cpu_memory_MB": 2552.312,
  "peak_gpu_0_memory_MB": 1837,
  "training_duration": "02:15:43",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.08730281049141247,
  "training_cpu_memory_MB": 2552.312,
  "training_gpu_0_memory_MB": 1837,
  "validation_BLEU": 0.5593781049121032,
  "validation_exprate": 0.23429541595925296,
  "validation_loss": 1.2164559353579272,
  "best_validation_BLEU": 0.5533631088763901,
  "best_validation_exprate": 0.2354272778720996,
  "best_validation_loss": 1.1377181824263152
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "multiscale",
        "encoder": {
            "type": 'multiscale',
            "encoder_type": 'resnet18',
            "encoder_height": 4,
            "encoder_width": 16,
            "pretrained": true,
            "custom_in_conv": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'multiscale',
                "attention": {
                    "type": 'image-captioning-attention',
                    "encoder_dim": 512, # Must be encoder dim of chosen encoder
                    "decoder_dim": 256, # Must be same as decoder's decoder_dim
                    "attention_dim": 256,
                    "doubly_stochastic_attention": false                    
                }
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```
### multi scale lstm encoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11930877 v66  
Results: 1.13; but bleu and exprate is a bit higher

```
{
  "best_epoch": 33,
  "peak_cpu_memory_MB": 2547.368,
  "peak_gpu_0_memory_MB": 1951,
  "training_duration": "02:30:05",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.17908455267953116,
  "training_cpu_memory_MB": 2547.368,
  "training_gpu_0_memory_MB": 1951,
  "validation_BLEU": 0.5652890622663765,
  "validation_exprate": 0.2597623089983022,
  "validation_loss": 1.1906401821084924,
  "best_validation_BLEU": 0.5524230597155718,
  "best_validation_exprate": 0.26485568760611206,
  "best_validation_loss": 1.1319132146534618
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "multiscale",
        "encoder": {
            "type": "multiscale-lstm",
            "encoder": {
                "type": 'multiscale',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512,
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning-decoder",
            "attention": {
                "type": 'multiscale',
                "attention": {
                    "type": 'image-captioning-attention',
                    "encoder_dim": 512, # Must be encoder dim of chosen encoder
                    "decoder_dim": 256, # Must be same as decoder's decoder_dim
                    "attention_dim": 256,
                    "doubly_stochastic_attention": true                    
                }
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### multiscale encoder and decoder
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments?scriptVersionId=11931072 v67  
Results: same loss worse metrics

```
{
  "best_epoch": 35,
  "peak_cpu_memory_MB": 2573.6,
  "peak_gpu_0_memory_MB": 1959,
  "training_duration": "02:39:19",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.25082658531304397,
  "training_cpu_memory_MB": 2573.6,
  "training_gpu_0_memory_MB": 1959,
  "validation_BLEU": 0.552428977897469,
  "validation_exprate": 0.26485568760611206,
  "validation_loss": 1.1610306138927873,
  "best_validation_BLEU": 0.5216430022519188,
  "best_validation_exprate": 0.2484436898698359,
  "best_validation_loss": 1.1368801502494124
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "math-dataset",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "math"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "multiscale",
        "encoder": {
            "type": "multiscale-lstm",
            "encoder": {
                "type": 'multiscale',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512,
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "msa-decoder",
            "attention": {
                "type": 'multiscale',
                "attention": {
                    "type": 'image-captioning-attention',
                    "encoder_dim": 512, # Must be encoder dim of chosen encoder
                    "decoder_dim": 256, # Must be same as decoder's decoder_dim
                    "attention_dim": 256,
                    "doubly_stochastic_attention": true                    
                }
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

### lstm lr 0.1
Kernel: https://www.kaggle.com/bkkaggle/math-recognition-experiments/data?scriptVersionId=11977575 v75  
Results: Good!

```
{
  "best_epoch": 16,
  "peak_cpu_memory_MB": 2474.772,
  "peak_gpu_0_memory_MB": 1441,
  "training_duration": "01:41:13",
  "training_start_epoch": 0,
  "training_epochs": 39,
  "epoch": 39,
  "training_loss": 0.5624248934277581,
  "training_cpu_memory_MB": 2474.772,
  "training_gpu_0_memory_MB": 1441,
  "validation_BLEU": 0.645596960102569,
  "validation_exprate": 0.40860215053763443,
  "validation_loss": 1.6618222970146317,
  "best_validation_BLEU": 0.6203121678315959,
  "best_validation_exprate": 0.4012450481041313,
  "best_validation_loss": 1.4731372686119768
}
```
```
%%writefile config.json
{
    "dataset_reader": {
        "type": "CROHME",
        "root_path": "./2013",
        "height": 128,
        "width": 512,
        "lazy": true,
        "subset": false,
        "tokenizer": {
            "type": "latex"
        }
    },
    "train_data_path": "train.csv",
    "validation_data_path": "val.csv",
    "model": {
        "type": "image-captioning",
        "encoder": {
            "type": "lstm",
            "encoder": {
                "type": 'backbone',
                "encoder_type": 'resnet18',
                "encoder_height": 4,
                "encoder_width": 16,
                "pretrained": true,
                "custom_in_conv": false
            },
            "hidden_size": 512, # Must be encoder dim of chosen encoder
            "layers": 1,
            "bidirectional": false
        },
        "decoder": {
            "type": "image-captioning",
            "attention": {
                "type": 'image-captioning',
                "encoder_dim": 512, # Must be encoder dim of chosen encoder
                "decoder_dim": 256, # Must be same as decoder's decoder_dim
                "attention_dim": 256,
                "doubly_stochastic_attention": true
            },
            "embedding_dim": 256,
            "decoder_dim": 256
        },
        "max_timesteps": 75,
        "beam_size": 10
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys":[["label", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.1,
            "momentum": 0.9
        },
#         "validation_metric": "+BLEU",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5
        },
        "num_serialized_models_to_keep": 1,
        "summary_interval": 10,
        "histogram_interval": 100,
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
    },
    "vocabulary": {
        "min_count": {
            'tokens': 10
        }
#         "directory_path": "/path/to/vocab"
    },
}
```

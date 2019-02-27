# math-expression-recognition

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

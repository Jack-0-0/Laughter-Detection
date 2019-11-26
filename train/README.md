# Train
network_train.py trains the laughter network and saves the results.

## Requirements
The following packages are needed:
* Python 3.4.8
* Tensorflow-gpu 1.12.0
* Keras 2.2.4
* numpy 1.16.2
* matplotlib 2.2.4
* cuda 9.0

## Datasets
Train, test and validation datasets for each language can be downloaded at [link to be added].

Alternatively they can be created as described in the preprocessing section [here](/preprocessing).

## Usage
Open network_train.py and change the variables under the settings section at the top of the file. For example, to train the network on 8,000 Hz Chinese data on GPU 2, the settings would be:

```python
########################
####### Settings ####### 
########################

# set path for training data
# if more than one language data, use np.vstack
# if using only one language data, set train = np.load(dataset_path)
# the network will use variable train
train = np.load('ch_train_6_6_64_ds.npy')

# set path for validation data
# if more than one language data, use np.vstack
# if using only one language data, set val = np.load(dataset_path)
# the network will use variable val
val = np.load('ch_val_6_6_64_ds.npy')

# set test data paths, these should not change between experiments 
# all networks are tested on all three language test sets
test_de = np.load('de_test_6_6_64_ds.npy')
test_ch = np.load('ch_test_6_6_64_ds.npy')
test_en = np.load('en_test_6_6_64_ds.npy')

# choose which GPU to use
gpu = "2"

# set number of mels
# set this to 64 for 22,050 Hz experiments or 45 for 8,000 Hz experiments
mels = 45

# set save folder identifier
# save folder will be named in format saved_models/date_time_filename_sfid 
sfid = 'CH_mels_' + str(mels) 

########################
```
To train the network on 22,050 Hz English and German data on GPU 3, the settings would be:

```python
########################
####### Settings ####### 
########################

# set path for training data
# if more than one language data, use np.vstack
# if using only one language data, set train = np.load(dataset_path)
# the network will use variable train
train_de = np.load('de_train_6_6_64_ds.npy')
train_en = np.load('en_train_6_6_64_ds.npy')
train = np.vstack((train_en, train_de))

# set path for validation data
# if more than one language data, use np.vstack
# if using only one language data, set val = np.load(dataset_path)
# the network will use variable val
val_de = np.load('de_val_6_6_64_ds.npy')
val_en = np.load('en_val_6_6_64_ds.npy')
val = np.vstack((val_en, val_de))

# set test data paths, these should not change between experiments 
# all networks are tested on all three language test sets
test_de = np.load('de_test_6_6_64_ds.npy')
test_ch = np.load('ch_test_6_6_64_ds.npy')
test_en = np.load('en_test_6_6_64_ds.npy')

# choose which GPU to use
gpu = "3"

# set number of mels
# set this to 64 for 22,050 Hz experiments or 45 for 8,000 Hz experiments
mels = 64

# set save folder identifier
# save folder will be named in format saved_models/date_time_filename_sfid 
sfid = 'DE_EN_mels_' + str(mels) 

########################
```
Save network_train.py with required settings and then run the file.

When the model is finished training, a folder will be created which contains:
* Plots of ROC AUC for each language test set
* F1, TPR, FPR, TNR, FNR scores for each language test set and validation set
* Network architecture
* Plots of loss and accuracy for each epoch
* History file
* Network save file which can be used for later evaluation or further training

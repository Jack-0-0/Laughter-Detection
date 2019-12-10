import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pickle
from keras.models import Model 
from keras.layers import Input
from keras.layers import MaxPooling2D, Activation, Dropout
from keras.layers import Reshape, Permute, multiply
from keras.layers import Conv2D, GRU, TimeDistributed, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_curve, auc
from keras import backend as K
import datetime

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


def create_model(filters, gru_units, dropout, bias, mels):
    """Create convolutional recurrent model.

    # Arguments
        filters: number of filters in the convolutional layers.
        gru_units: number of gru units in the GRU layers.
        dropout: neurons to drop out during training. Values of between 0 to 1.
        bias: set to True or False. Should be False when using BatchNorm, True when not.

    # Returns
        Keras functional model which can then be compiled and fit.
    """
    inp = Input(shape=(259, mels, 1))
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(inp)
    x = MaxPooling2D(pool_size=(1,5))(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(x)
    x = MaxPooling2D(pool_size=(1,2))(x)

    x = Reshape((x_train.shape[-3], -1))(x)

    x = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=bias), merge_mode='mul')(x)
    
    x = TimeDistributed(Dense(512, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    x = TimeDistributed(Dense(256, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    x = TimeDistributed(Dense(128, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inp], outputs=output)
    return model


def save_folder(date_time, sfid):
    """Create save folder and return the path.

    # Arguments
        date_time: Current time as per datetime.datetime.now()

    # Creates
    	directory at save_folder location, if it does not exist already.

    # Returns
        path to save folder.
    """
    date_now = str(date_time.date())
    time_now = str(date_time.time())
    sf = "saved_models/" + date_now + "_" + time_now + "_" + os.path.basename(__file__).split('.')[0] + '_' + sfid
    if not os.path.isdir(sf):
        os.makedirs(sf)
    return sf

        
def save_model(save_folder):
    """Saves model and history file.

    # Arguments
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves model and history.
    """ 
    model.save(save_folder + '/savedmodel' + '.h5')
    with open(save_folder +'/history.pickle', 'wb') as f_save:
        pickle.dump(model_fit.history, f_save)


def plot_accuracy(model_fit, save_folder):
    """Plot the accuracy during training for the train and val datasets.

    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation accuracy at each epoch.
    """ 
    train_acc = model_fit.history['binary_accuracy']
    val_acc = model_fit.history['val_binary_accuracy']
    epoch_axis = np.arange(1, len(train_acc) + 1)
    plt.title('Train vs Validation Accuracy')
    plt.plot(epoch_axis, train_acc, 'b', label='Train Acc')
    plt.plot(epoch_axis, val_acc,'r', label='Val Acc')
    plt.xlim([1, len(train_acc)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_acc) / 10) + 0.5)))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/accuracy.png')
    plt.show()
    plt.close()
    

def plot_loss(model_fit, save_folder):
    """Plot the loss during training for the train and val datasets.

    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation loss at each epoch.
    """ 
    train_loss = model_fit.history['loss']
    val_loss = model_fit.history['val_loss']
    epoch_axis = np.arange(1, len(train_loss) + 1)
    plt.title('Train vs Validation Loss')
    plt.plot(epoch_axis, train_loss, 'b', label='Train Loss')
    plt.plot(epoch_axis, val_loss,'r', label='Val Loss')
    plt.xlim([1, len(train_loss)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_loss) / 10) + 0.5)))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/loss.png')
    plt.show()
    plt.close()


def plot_ROC(model, x_test, y_test, save_folder, lang):
    """Plot the ROC with AUC value.

    # Arguments
        model: model after training.
        x_test: inputs to the network for testing.
        y_test: actual outputs for testing.
        save_folder: path for directory to save model and related history and metrics.
        lang: language of dataset. 

    # Output
        saves plots of ROC.
    """ 
    predicted = model.predict(x_test).ravel()
    actual = y_test.ravel()
    fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Test ROC AUC ' + lang)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_folder + '/ROC_' + lang + '.png')
    plt.show()
    plt.close()


def metrics(x, y, save_folder, threshold, ds_name):
    """Calculate the TPR, TNR, FPR, FNR and F1 score.

    # Arguments
        x: inputs to the network.
        y: actual outputs.
        save_folder: path for directory to save model and related history and metrics.
        threshold: values greater than threshold get set to 1, values less than or
                   equal to the threshold get set to 0.
        df_name: identifier for text file.

    # Output
        saves True Positive Rate, True Negative Rate, False Positive Rate, False Negative Rate
        dependent on threshold.
    """
    predicted = model.predict(x)
    predicted[predicted > threshold] = 1
    predicted[predicted <= threshold] = 0
    actual = y
    TP = np.sum(np.logical_and(predicted == 1, actual == 1))
    FN = np.sum(np.logical_and(predicted == 0, actual == 1))
    TN = np.sum(np.logical_and(predicted == 0, actual == 0))
    FP = np.sum(np.logical_and(predicted == 1, actual == 0))
    TPR  = TP / (TP + FN + 1e-8)
    TNR  = TN / (TN + FP + 1e-8)
    FPR = FP / (FP + TN + 1e-8)
    FNR = FN / (FN + TP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TPR
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    metrics_dict = {'TPR': np.round(TPR, 3),
                    'TNR': np.round(TNR, 3),
                    'FPR' : np.round(FPR, 3),
                    'FNR' : np.round(FNR, 3),
                    'F1 Score' : np.round(F1, 3)
                   }
    with open(save_folder + '/' + ds_name + '_metrics.txt', 'w') as f:
        f.write(str(metrics_dict))


def save_arch(model, save_folder):
    """Saves the network architecture as a .txt file.

    # Arguments
        model: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves network architecture.
    """
    with open(save_folder + '/architecture.txt','w') as a_save:
        model.summary(print_fn=lambda x: a_save.write(x + '\n'))


def find_mean_stdd(dataset):
    """Find mean and standard deviation of the dataset.

    # Arguments
        dataset: dataset in format (id, spectro, label).
        where spectro is in format (n, timesteps, mel bands, 1).
        
    # Returns
        mean: mean for each mel band across the dataset.
        stdd: standard deviation for each mel band across the dataset.
    """
    x = dataset[:, 1]
    x = np.stack(x) # reshape to (n, mel bands, timesteps)
    mean = x.mean(axis=(0, 2)) # mean in shape (mel bands, )
    mean = np.expand_dims(mean, axis=1) # reshape mean to (mel bands, 1)
    stdd = x.std(axis=(0, 2)) # std in shape (mel bands, )
    stdd = np.expand_dims(stdd, axis=1) # reshape stdd to (mel bands, 1)
    return mean, stdd


def normalise_and_reformat(dataset, mean, stdd):
    """Normalise data based on training data and reformat into suitable format.

    # Arguments
        dataset: dataset in format (id, spectro, label)
        mean: mean for each mel band across the dataset.
        stdd: standard deviation for each mel band across the dataset.
        
    # Returns
        x: spectros normalised across each mel band in format (n, timesteps, mel bands, 1)
        y: labels in format (n, timesteps, 1)
    """
    x = dataset[:, 1] 
    x = np.stack(x) # reshape to (n, mel bands, timesteps)
    x = (x - mean) / (stdd + 1e-8) # normalise so mean is equal to zero and variance equal to 1
    x = np.expand_dims(np.moveaxis(x, 1, -1), axis=3) # reformat x to (n, timesteps, mel bands, 1)  
    y = dataset[:, 2] 
    y = np.expand_dims(np.moveaxis(np.stack(y), 1, -1), axis=2) # reformat y to (n, timesteps, 1)
    return x, y


# GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
tf.Session(config=config)

# reformat and normalise datasets
mean, stdd = find_mean_stdd(train)

x_train, y_train = normalise_and_reformat(train, mean, stdd)
x_val, y_val = normalise_and_reformat(val, mean, stdd)

x_test_en, y_test_en = normalise_and_reformat(test_en, mean, stdd)
x_test_ch, y_test_ch = normalise_and_reformat(test_ch, mean, stdd)
x_test_de, y_test_de = normalise_and_reformat(test_de, mean, stdd)

# reduce number of mel bands for spectrograms, to represent lower sample rate
x_train = x_train[:, :, :mels, :]
x_val = x_val[:, :, :mels, :]
x_test_en = x_test_en[:, :, :mels, :]
x_test_ch = x_test_ch[:, :, :mels, :]
x_test_de = x_test_de[:, :, :mels, :]

# create network
model = create_model(filters=128, gru_units=128, dropout=0.5, bias=True, mels=mels)
print(model.summary())
adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

# train network
model_fit = model.fit(x_train, y_train, epochs=2500, batch_size=192,
                      validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr_plat])

# save network, training info, performance measures
date_time = datetime.datetime.now()
sf = save_folder(date_time, sfid)
save_model(sf)
plot_accuracy(model_fit, sf)
plot_loss(model_fit, sf)
plot_ROC(model, x_test_en, y_test_en, sf, 'EN')
plot_ROC(model, x_test_ch, y_test_ch, sf, 'CH')
plot_ROC(model, x_test_de, y_test_de, sf, 'DE')
save_arch(model, sf)
metrics(x_val, y_val, sf, 0.5, 'val')
metrics(x_test_en, y_test_en, sf, 0.5, 'en_test')
metrics(x_test_ch, y_test_ch, sf, 0.5, 'ch_test')
metrics(x_test_de, y_test_de, sf, 0.5, 'de_test')

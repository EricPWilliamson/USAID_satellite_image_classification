"""
This script based on the blog post "Building powerful image classification models using very little data"
    from blog.keras.io.

*First major run: accuracy was static over epochs at only 0.0038 (val) and 0.0039 (train). Evidently, the model was
  predicting 'settlement' for every image.
*After changing the output layer to 3 neurons and changing loss to categorical_crossentropy, results were poor, but realistic.
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import Callback
import pickle

def count_subdir_files(folderpath):
    tot = 0
    for _,_,files in os.walk(folderpath):
        tot += sum(1 for file in files)
    return tot


DESC = 'bottleneck_rgb_png'

# dimensions of our images.
img_width, img_height = 200, 200

top_model_weights_path = 'bottleneck_usaid_model_1.h5'
train_data_dir = 'USAID_africa/simple_RGB'
# validation_data_dir = 'USAID_africa/simple_RGB/validation'
# train_data_dir = 'USAID_africa/simple_RGB_lite'
epochs = 50
batch_size = 8

# #Count number of images available:
# nb_train_samples = count_subdir_files(train_data_dir)
# nb_validation_samples = count_subdir_files(validation_data_dir)


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.25)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    t_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        subset='training')
    # bottleneck_features_train = model.predict_generator(t_generator, t_generator.samples // batch_size)
    bottleneck_features_train = model.predict_generator(t_generator, t_generator.samples)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    np.save(open('bottleneck_labels_train.npy', 'wb'), t_generator.classes)

    # v_generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    v_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width,img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        subset='validation')
    # bottleneck_features_validation = model.predict_generator(v_generator, v_generator.samples // batch_size)
    bottleneck_features_validation = model.predict_generator(v_generator, v_generator.samples)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    np.save(open('bottleneck_labels_validation.npy', 'wb'), v_generator.classes)



def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = to_categorical(np.load(open('bottleneck_labels_train.npy', 'rb')))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = to_categorical(np.load(open('bottleneck_labels_validation.npy', 'rb')))

    #Define sample weights to emphasize the underrepresented classes:
    sample_weight = np.dot(train_labels, [9.3, 229.5, 1.0])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        sample_weight=sample_weight,
                        verbose=2)
    model.save_weights(top_model_weights_path)
    return model, history

# save_bottlebeck_features() #!!can skip after 1st run
model,history = train_top_model()
###Save model and pickle history:
print('Saving model and history...')
model.save('./USAID_africa/Models/'+DESC+'_model.h5')
hist = history.history
# hist['fit_times'] = time_callback.times
filename = './USAID_africa/Models/'+DESC+'_history.pkl'
with open(filename, 'wb') as pfile:
    pickle.dump(hist, pfile, pickle.HIGHEST_PROTOCOL)

print('done')


# Plot training & validation accuracy values
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

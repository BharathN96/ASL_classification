import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

vgg_model = tf.keras.models.load_model("vgg16_model")

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_data = train_datagen.flow_from_directory("Alphabets_split_1/train",
                                               target_size=(32, 32),
                                               class_mode='categorical',
                                               batch_size=8, )


def get_asl_class(hand_image):
    test_image = image.smart_resize(hand_image, (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    vgg_res = vgg_model.predict(test_image / 255.)
    vgg_prediction = list(train_data.class_indices.keys())[np.argmax(vgg_res)]

    return vgg_prediction

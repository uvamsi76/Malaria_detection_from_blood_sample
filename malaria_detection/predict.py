import cv2
from PIL import Image
import argparse
import warnings
import tensorflow as tf
import os
import absl.logging


def predict(image_path):
    saved_model = tf.keras.models.load_model('malaria_detection/models/Lenet_model.h5')
    image=cv2.imread(image_path)
    image=tf.image.resize([image],(224,224))
    output=saved_model.predict(image)
    output=0 if output<0.5 else 1
    print()
    print(output)
    print()
    return output

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--path", type=str, help="path for input image")
    args = parser.parse_args()
    path='data/single_prediction/Parasitised.png'
    if(args.path):
        path=args.path
    else:
        print('taking default path for image')
    predict(path)
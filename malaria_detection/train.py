import tensorflow as tf
from malaria_detection.train_util import train, lenet_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("--epochs", type=int, help="number of epochs")
parser.add_argument("--datadir", type=str, help="path for parasitised training images")
args = parser.parse_args()

epochs=10
data_dir='data/training_set'
if(args.epochs):
    epochs=args.epochs
if(args.datadir):
    data_dir=args.datadir

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",           
    label_mode="int",            
    batch_size=32,               
    image_size=(224, 224),       
    shuffle=True,                
    seed=42,                     
    validation_split=0.2,        
    subset="training"            
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"           
)
model=lenet_model
model= train(train_ds,val_ds,model,epochs,model_version="1.0.0",lr=1e-3)
print(model.summary())
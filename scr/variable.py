DATASET=200
TRAIN=190
VAL=5
TEST=5

DATASET_PATH='../training'
SHUFFLE_DATA=True
EPOCHS=50
LEARNING_RATE=0.0005

NUMBER_OF_CHANNELS = 1
CLASSES = 4 #4 classes: ventricule droit & gauche + orreillete + backgroude

DEEP_UNET=4 # profondeur du Unet

SAVE_BEST_ONLY=True
METRIC='IOU_CLASS' # choisir la métric et la loss entre 'IOU' et 'IOU_CLASS's


from model_BERT import utils, config
import joblib
import os

os.makedirs(config.MODEL_SAVE_PATH,exist_ok=True)

X_train, y_train = utils.load_data(config.TRAIN_FILE, fit_label_encoder=True)
joblib.dump(utils.label_encoder,os.path.join(config.MODEL_SAVE_PATH,"label_encoder.pkl"))

print("Label encoder saved successfully!")
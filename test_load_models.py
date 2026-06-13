import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5 = os.path.join(BASE_DIR, 'keras_model.h5')
SAVEDMODEL_DIR = os.path.join(BASE_DIR, 'model.savedmodel')

def try_load(model_path):
    print(f"\nTentando carregar: {model_path}")
    is_h5 = str(model_path).lower().endswith(('.h5', '.keras'))
    is_dir = os.path.isdir(model_path)
    try:
        if is_h5:
            m = load_model(model_path, compile=False)
        elif is_dir:
            TFSMLayer = None
            try:
                from keras.layers import TFSMLayer  # type: ignore
            except Exception:
                try:
                    from tensorflow.keras.layers import TFSMLayer  # type: ignore
                except Exception:
                    TFSMLayer = None

            if TFSMLayer is None:
                print("TFSMLayer não disponível — pule o carregamento do SavedModel.")
                return False

            model = tf.keras.Sequential([
                tf.keras.Input(shape=(224, 224, 3)),
                TFSMLayer(model_path, call_endpoint='serving_default')
            ])
            m = model
        else:
            m = load_model(model_path, compile=False)

        x = np.zeros((1, 224, 224, 3), dtype=np.float32)
        try:
            p = m.predict(x)
            print("Predict OK — saída com shape:", getattr(p, 'shape', type(p)))
        except Exception as e:
            print("Carregado, mas predição falhou:", e)
        return True
    except Exception as e:
        print("Falha ao carregar:", e)
        return False

def main():
    print("Diretório base:", BASE_DIR)
    if os.path.exists(MODEL_H5):
        try_load(MODEL_H5)
    else:
        print("Modelo .h5 não encontrado em:", MODEL_H5)

    if os.path.isdir(SAVEDMODEL_DIR):
        try_load(SAVEDMODEL_DIR)
    else:
        print("SavedModel não encontrado em:", SAVEDMODEL_DIR)

if __name__ == '__main__':
    main()

import discord
from discord.ext import commands
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'keras_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

def get_class(model_path, labels_path, image_path):
    # Carregar o modelo
    model = None
    # Se for arquivo .h5/.keras, usar load_model; se for SavedModel (diretório), tentar TFSMLayer
    is_h5 = str(model_path).lower().endswith(('.h5', '.keras'))
    is_savedmodel_dir = os.path.isdir(model_path) or str(model_path).lower().endswith('savedmodel')

    if is_h5:
        model = load_model(model_path, compile=False)
    elif is_savedmodel_dir:
        # Tentar importar TFSMLayer (prefere standalone keras, senão tf.keras)
        TFSMLayer = None
        try:
            from keras.layers import TFSMLayer  # type: ignore
        except Exception:
            try:
                from tensorflow.keras.layers import TFSMLayer  # type: ignore
            except Exception:
                TFSMLayer = None

        if TFSMLayer is None:
            raise RuntimeError("TFSMLayer não disponível. Instale 'keras' compatível ou atualize TensorFlow/Keras para carregar SavedModel.")

        # Construir um modelo sequencial de entrada seguido pelo TFSMLayer para inferência
        input_shape = (224, 224, 3)
        try:
            model = tf.keras.Sequential([
                tf.keras.Input(shape=input_shape),
                TFSMLayer(model_path, call_endpoint='serving_default')
            ])
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar SavedModel via TFSMLayer: {e}")
    else:
        # Fallback: tentar load_model e deixar erro natural propagar
        model = load_model(model_path, compile=False)
    
    # Carregar os labels
    with open(labels_path, 'r') as f:
        labels = [line.split(' ', 1)[1] if ' ' in line else line.strip() for line in f]
    
    # Abrir e pré-processar a imagem
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Assumindo tamanho comum; ajuste se necessário
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = np.expand_dims(image_array, axis=0)
    
    # Fazer predição
    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions)
    
    return labels[class_idx]

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Olá {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            await ctx.send(get_class(model_path=MODEL_PATH, labels_path=LABELS_PATH, image_path=f"./{attachment.filename}"))
    else:
        await ctx.send("Você esqueceu de enviar a imagem :(")
bot.run("MTUxMjgxMjIyMTUzNTc1MjM5NA.GJXxEs.FzSSDPKksRxTKPT0MlJwXSBPY6VpGPC4OzkHWc")
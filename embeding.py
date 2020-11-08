from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import pandas as pd

def load_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert("RGB")

    return asarray(image)


def load_faces(directory_src):
    faces = list()

    for filename in listdir(directory_src):
        path = directory_src + filename

        try:
            faces.append(load_face(path))
        except:
            print: ("erro na imagem {}".format(path))

    return faces


def load_fotos(directory_src):
    x, y = list(), list()

    for subdir in listdir(directory_src):
        path = directory_src + subdir + '\\'

        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]
        print('>carregadas %d faces da classe: %s' % (len(faces), subdir))

        x.extend(faces)
        y.extend(labels)

    return asarray(x), asarray(y)

trainX, trainy = load_fotos(directory_src = "C:\\Users\\caio\\Pictures\\fotos_IA\\faces\\")

from tensorflow.keras.models import load_model
model = load_model ('facenet_keras.h5')
model.summary()

def get_embedding(model, face_pixel):
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixel = (face_pixel - mean)/std
    samples = expand_dims(face_pixel, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
newTrainX.shape
df = pd.DataFrame(data=newTrainX)
df
df['target'] = trainy

df.to_csv('caio.csv')
from sklearn.utils import shuffle
X,y = shuffle(newTrainX, trainy, random_state=0)

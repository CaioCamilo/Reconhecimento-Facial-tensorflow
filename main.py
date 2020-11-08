# codigo para cortar as fotos
# o que quer é refletir as facas na pasta "face", essas pastas ja estão criadas em fotos
# adicinar uma biblioteca que reconheça faces
from mtcnn import MTCNN  # para extrair as faces
from PIL import Image  # para manipular imagem
from os import listdir  # para listar diretorio ou pastas
from os.path import isdir  # para saber se o que foi lido é um diretorio ou não
from numpy import asarray  # para converter uma imagem pil em array
detector = MTCNN()  # para enxergar as faces dentro de uma imagem

def  load_face(arquivo):  # função para extrair as faces, em "arquivo" é a filename
    img = Image.open(arquivo)  # o "arquivo deve ser o endereço completo path
    img = img.convert('RGB')  # converter a imagem em rgb
    return asarray(img)
def load_faces(directory_src):
    faces = []
    for filename in listdir(directory_src):
        try:
            faces.append(load_face(directory_src + filename))
        except:
            print("Deu ruim mano! Olha em ".format(str(directory_src + filename)))
    return faces

def load_fotos(directory_src):
    X, y = list(), list()
    for subdir in listdir(directory_src):
        path = directory_src + subdir + "\\"
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('Carregamos %d faces do(a): %s'%(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

trainX, trainy = load_fotos(directory_src='C:\\Users\caio\\Pictures\\fotos_IA\\faces\\')


def extrair_face(arquivo, size=(160, 160)):  # função para extrair as faces, em "arquivo" é a filename
    img = Image.open(arquivo)  # o "arquivo deve ser o endereço completo path
    img = img.convert('RGB')  # converter a imagem em rgb
    array = asarray(img)  # convertar a imagem em matriz pois o detector mtcnn le o arquivo em numpy
    results = detector.detect_faces(array)  # aqui ele vai dar o resultado das imagens e sobre tb as distancias dos orgão faciais
    try:
        x1, y1, width, height = results[0]['box']  # vai retornar o ponto mais alto e a esquerda da face, largura e altura
    except:
        x1, y1, width, height = 1,1,1,1

    x2 = x1 + width
    y2 = y1 + height  # com isso será formado o quadrado no rosto da pessoa

    # agora vamos fazer uma sub imagem da imagem ja cortada
    face = array[y1:y2, x1:x2]
    # para transformar a imagem em um quadradro, vamos voltar isso para o PIL
    image = Image.fromarray(face)
    image = image.resize(size)
    return image

def load_fotos(directory_src, directory_target):
   for filename in listdir(directory_src):
       path = directory_src + filename
       path_tg = directory_target + filename
       print(path_tg)
       face = extrair_face(path)
       face.save(path_tg)

# precisamos abrir os diretorios e os subdiretorios para reconhecer as faces
def load_dir(directory_src, directory_target):  # src é o diretorio fotos e target o diretorio faces
    for subdir in listdir(directory_src):
        path = directory_src + subdir + "\\"
        path_tg = directory_target + subdir + "\\"
        if not isdir(path):
            continue
        load_fotos(path, path_tg)

if __name__ == '__main__':
    load_dir('C:\\Users\\caio\\Pictures\\fotos_IA\\fotos\\',
             'C:\\Users\\caio\\Pictures\\fotos_IA\\faces\\')
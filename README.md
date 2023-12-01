# TCC-Deep-Learning-Mackenzie
Código utilizado no Trabalho de Conclusão de Curso em Engenharia de Produção na Universidade Presbiteriana Mackenzie. 

A definição de Inteligência Artificial consiste na capacidade que as máquinas têm de realizar tarefas complexas que antes eram associadas apenas a seres humanos. Esse campo é objeto de estudo acadêmico e busca a autonomia das funções executadas por tais mecanismos. Suas técnicas incluem aprendizado de máquina (machine learning), redes neurais (neural networks), algoritmos genéticos (genetic algorithms) e sistemas especialistas (expert systems), entre outras. Este trabalho possui o objetivo de estudar os diferentes sistemas de Inteligência Artificial, focando em Redes Neurais aplicadas em Deep Learning, assim como o desenvolvimento de uma aplicação que utilize esses sistemas. O escopo do projeto analisará a possibilidade de uso da linguagem Python e das bibliotecas "TensorFlow" e "Keras" para criar  modelos com topologia não-linear, camadas compartilhadas e vários inputs e outputs desejados. A importância da investigação presente neste trabalho reside na relação entre os avanços na criação de inteligências artificiais, suas aplicações e um estudo sobre a modulação e funcionamento relacionado a redes neurais e aprendizado profundo. O objetivo é investigar, analisar e exemplificar um modelo de rede neural, oferecendo uma compreensão aprofundada das técnicas e conceitos subjacentes. Ao aplicar modelos de rede neural em complicações de manufatura, os engenheiros de produção podem aprimorar a eficiência operacional, reduzir custos e minimizar desperdícios. Os resultados indicam uma alta precisão dos testes realizados no modelo (99,17%), conseguindo classificar com grande acurácia dígitos numéricos em um intervalo de [0, 9].Além disso, a análise de dados avançada oferecida pelo Deep Learning possibilita uma visão mais precisa das operações de produção, permitindo a identificação de gargalos e a implementação de melhorias substanciais.

**Palavras-chave:** Inteligência Artificial. Deep Learning. Redes Neurais.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizando os valores pixel em uma faixa entre 0 e 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Remodelando as imagens para (28, 28, 1) para entrada CNN
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Construindo o modelo de Rede Neural Convolucional
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), #achatamento
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # regularizando
    layers.Dense(10, activation='softmax')
])

# Compilando o modelo, especificando a funcao de perda, o otimizador e as metricas
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo usando os dados de treinamento
history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))

# Avaliação de desempenho do modelo no conjunto de dados de teste
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisão do teste: {test_acc}')

# Utilizando o modelo treinado para fazer previsões sobre dados novos e não vistos
predictions = model.predict(test_images)

import matplotlib.pyplot as plt

# Função para exibir o dígito e seu rótulo previsto
def plot_image(i, predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = tf.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f'Predicted: {predicted_label} ({100 * tf.reduce_max(predictions_array):.2f}%)', color=color)

# Traçando as primeiras imagens de teste e suas previsões
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels[i], test_images[i])

plt.tight_layout()
plt.show()

# %%
# Matriz de Confusão

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

predictions = model.predict(test_images)
final_predictions = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, final_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

import cv2
# Carregando e pré-processando a image inserida
input_image = cv2.imread('5peq.png', cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (28, 28))
input_image = input_image / 255.0  # Normalizando valores de pixel
input_image = input_image.reshape(1, 28, 28, 1)  # Remodelando para entrada do modelo

import numpy as np
# Usando o modelo treinado para realizar predições
predictions = model.predict(input_image)
predicted_label = np.argmax(predictions)

print(f'Predicted label: {predicted_label}')

# Exibindo a imagem de entrada
import matplotlib.pyplot as plt
plt.imshow(input_image.reshape(28, 28), cmap='gray')
plt.title(f'Previsão numérica: {predicted_label}')
plt.show()

# TCC-Deep-Learning-Mackenzie
Código utilizado no Trabalho de Conclusão de Curso em Engenharia de Produção na Universidade Presbiteriana Mackenzie. 

A definição de Inteligência Artificial consiste na capacidade que as máquinas têm de realizar tarefas complexas que antes eram associadas apenas a seres humanos. Esse campo é objeto de estudo acadêmico e busca a autonomia das funções executadas por tais mecanismos. Suas técnicas incluem aprendizado de máquina (machine learning), redes neurais (neural networks), algoritmos genéticos (genetic algorithms) e sistemas especialistas (expert systems), entre outras. Este trabalho possui o objetivo de estudar os diferentes sistemas de Inteligência Artificial, focando em Redes Neurais aplicadas em Deep Learning, assim como o desenvolvimento de uma aplicação que utilize esses sistemas. O escopo do projeto analisará a possibilidade de uso da linguagem Python e das bibliotecas "TensorFlow" e "Keras" para criar  modelos com topologia não-linear, camadas compartilhadas e vários inputs e outputs desejados. A importância da investigação presente neste trabalho reside na relação entre os avanços na criação de inteligências artificiais, suas aplicações e um estudo sobre a modulação e funcionamento relacionado a redes neurais e aprendizado profundo. O objetivo é investigar, analisar e exemplificar um modelo de rede neural, oferecendo uma compreensão aprofundada das técnicas e conceitos subjacentes. Ao aplicar modelos de rede neural em complicações de manufatura, os engenheiros de produção podem aprimorar a eficiência operacional, reduzir custos e minimizar desperdícios. Os resultados indicam uma alta precisão dos testes realizados no modelo (99,17%), conseguindo classificar com grande acurácia dígitos numéricos em um intervalo de [0, 9].Além disso, a análise de dados avançada oferecida pelo Deep Learning possibilita uma visão mais precisa das operações de produção, permitindo a identificação de gargalos e a implementação de melhorias substanciais.

**Palavras-chave:** Inteligência Artificial. Deep Learning. Redes Neurais.
print(f'Predicted label: {predicted_label}')

# Exibindo a imagem de entrada
import matplotlib.pyplot as plt
plt.imshow(input_image.reshape(28, 28), cmap='gray')
plt.title(f'Previsão numérica: {predicted_label}')
plt.show()

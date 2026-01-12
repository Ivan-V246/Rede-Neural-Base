# Rede Neural Básica
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-Operações_Matriciais-brightgreen?logo=NUmpy&logoColor=white)
![Git](https://img.shields.io/badge/Git-Versionamento-orange?logo=git&logoColor=white)


Este projeto tem como objetivo a criação de um **modelo de rede neural** personalizável em Python. Implementado para fins de estudo, tanto da implementação prática quanto do impacto de diferentes HiperParâmetros.

---

## 📁 Estrutura do projeto

O projeto é dividido em três arquivos principais:

- **Rede.py** - Implementa a class Rede_Neural, que possui os métodos de FeedFoward, Backprop e Learn;

- **Gera_Dados.py** - Implementa um gerador de dataset para treinamento da Rede Neural, armazenando no formato **Entrada | Saída** no arquivo **Dados.txt**;

- **IA.py** - Permite ao usúario escolher os HiperParâmetros de sua escolha pra Rede Neural, como **Quantidade de Camadas**, **Neurônios para cada camada**, **Taxa de Aprendizado** e **Rounds de treino por época**.

---

## 🛠️ Ferramentas Utilizadas
- **[Python](https://www.python.org/)** - Linguagem de programação principal do projeto.  
- **[Numpy](https://numpy.org/doc/)** - Biblioteca para cálculos matriciais eficientes.  
- **[Git](https://git-scm.com/)** - Versionamento e controle do código.  
---

## Como testar
Com python instalado:
```bash
    git clone https://github.com/Ivan-V246/Rede-Neural-Base.git
    cd Rede-Neural-Base/
    pip install -r requirements
    python Gera_Dados.py
    python RN.py
```

O programa RN.py irá instanciar a classe Rede_Neural, com os parâmetros definidos pelo usuário, e apresentar as **saídas esperadas** e a **saídas do modelo** para cada input do conjunto de treino, assim como também a margem de erro total daquela versão do modelo. 
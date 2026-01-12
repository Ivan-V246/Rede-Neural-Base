from Rede import Rede_Neural
import numpy as np
import ast

#Lê os dados de treino, armazenados em dados.txt
dataset = []

with open("Dados.txt", "r") as data:
    for linha in data:
        temp = (linha.strip().split('|'))

        entrada = np.array(ast.literal_eval(temp[0]))
        saida = np.array(ast.literal_eval(temp[1]))
        amostra = (entrada, saida)
        dataset.append(amostra)

#Recebe a informações da rede, quantidade de camadas, neurônios e taxa de aprendizado
layers = int(input("Insira a quantidade de camadas: (Mínima de três) \n"))
print()

camadas = []
for i in range (layers):
    a = int(input(f"Quantidade de neurônios da camada {i+1}: "))
    camadas.append(a)
print()


taxa = float(input("Insira a taxa de aprendizado requerida: \n"))
treino = int(input("Insira a quantidade de rounds de treino: \n"))

#Cria a rede
net = Rede_Neural(camadas)
cont = 0
while(1):

    print("O modelo foi treinado ", cont, " vezes.")
    erro = 0
    print("Resposta esperada >>>>> Resposta do modelo")
    for i in dataset:
        respM = net.feedforward(i[0])
        respE = i[1]
        temp = respM - respE

        print(respE, " >>>>> ", respM )

        erro+= (0.5* temp * temp)

    print()
    print("Margem de erro: ", erro)
    print()

    c = 1
    while(c <= treino):
        net.learn(dataset, taxa)
        c+=1
    cont += 1
    continuar = input("Deseja continuar? 0 para parar: ")
    print()
    if(continuar == '0'):
        break
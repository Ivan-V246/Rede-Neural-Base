import numpy as np

def sigmoid(z): #Z é um vetor
    return 1.0/(1.0+np.exp(-z)) #Função de Sigmoid que normaliza os valores

def sigmoid_prime(z): #Z é um vetor
    a = sigmoid(z)
    return a * (1-a) #FDerivada da função de Sigmoid, usada no cálculo das gradientes


class Rede_Neural(object):

    def __init__(self, sizes):
        self.quant_camadas = len(sizes) #Quantidade de camadas 

        self.sizes = sizes #Armazena quantos neurônios tem por camada

        #Parâmetros inicialmente aletórios

        self.vieses = [np.random.randn(y, 1) for y in sizes[1:]] #Cria os Vieses pra cada camada, Matriz com y linhas e 1 coluna

        self.pesos = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])] 
        
        #Cria os pesos de cada conexão, num formato de matriz; Cada posição dessas possui y linhas e x colunas;

        #pess[1] armazena os pesos da Segunda pra Terceira Camada; pesos[1][j][k] armazena o valor do K-Ésimo neurônio da segunda
        #pro J-Ésimo neurônio da terceira;


        self.entradas = [None] * self.quant_camadas 
        #Armazena os valores que chegam a cada camada de neurônio, sendo o último o valor final da rede 
        self.saidas = [None] * (self.quant_camadas-1)
        # Armazena os valores z = W·a + b (pré-ativação) de cada camada 

        self.grad_V = [None] *(self.quant_camadas-1) #Gradiente das Vieses
        self.grad_P = [None] *(self.quant_camadas-1) #Gradiente dos Pesos


    def feedforward(self, a): #Função por passar um valor de entrada (matriz a) pela rede neural

        i = 0
        for b, w in zip(self.vieses, self.pesos):
            self.entradas[i] = a
            self.saidas[i] = (np.dot(w, a)+b)
            a = sigmoid(self.saidas[i])
            i+=1

        self.entradas[i] = a
        return a
    
    def backprop(self, y): #Função responsável por calcular os gradientes de atualização de parâmetros
        
        self.grad_V[-1] = (self.entradas[-1] -y) * sigmoid_prime(self.saidas[-1]) 
        #Gradiente da camada de Saída: Derivada de: Função de perda(Normalizadora(Última saída));

        self.grad_P[-1] = self.grad_V[-1]*self.entradas[-2].T
        #Gradiente dos pesos que chegam na camada de saída: Mesmo dos vieses * Respostas que cada neurônio enviou;
        #Precisa ser transposta para inverter as dimensões do FeedFoward

        i = len(self.saidas)-2
        while(i > -1):
            self.grad_V[i] = np.dot((self.pesos[i+1].T), self.grad_V[i+1]) * sigmoid_prime(self.saidas[i])
            #O delta pra uma camada oculta é:

            #O produto escalar dos pesos que saem dessa camada (Pesos+1) (transposta) * Delta da camada da frente.
            #Pode pensar nesse valor como o Erro (Delta) da camada posterior retornando multiplicado pelos pesos correspodentes;

            #Isso * A derivada da Normalizadora pra cada valor gerado pelos neurônios 

            self.grad_P[i] = self.grad_V[i]*self.entradas[i].T
            #Gradiente dos pesos que chegam nessa camada
            #Delta dessa camada * Respostas que cada neurônio enviou (transposta)
            i-=1

    
    def learn(self, data, eta):


        for i, j in data:

            self.feedforward(i)
            #Passa o valor pela rede para registrar os valores de entradas e saídas das camadas.

            self.backprop(j)
            #Calcula os gradientes de atualização pra aquele valor de entrada

            for l in range(len(self.pesos)):
                self.vieses[l]  -= eta * self.grad_V[l]
                self.pesos[l] -= eta * self.grad_P[l]
                #Atualiza os parâmetros na direção descendente do Gradiente

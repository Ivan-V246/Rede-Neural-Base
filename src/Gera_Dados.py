dados = [None]*8
resp = [None]*8

for i in range(8):

    entrada = [None]*3
    bit = [None]*3

    for j in range(3):
        if (i | (1 << j)) == i:
            bit[j] = 1
            entrada[j] = [1.0]
        else:
            bit[j] = 0
            entrada[j] = [0.0]
    ans = (bit[0] and bit[1]) or bit[2]
    
    dados[i] = entrada
    resp[i] = [float(ans)]

with open("Dados.txt", "w") as data:
    for i, j in zip(dados, resp):
        data.write(str(i) + "|") 
        data.write(str(j) + "\n")
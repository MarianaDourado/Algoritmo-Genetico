from typing import List, Sequence
import random
import numpy as np
import matplotlib.pyplot as plt


Genoma = List[int] # Define Genoma como uma lista de inteiros
Populacao = List[Genoma] # Define população como uma lista de genomas
tam_genoma = 44 # Quantos bits tem o genoma
pop_size = 100 # Tamanho da população
num_ger = 100 # Número de Gerações


#---------------------------------- POPULAÇÃO INICIAL ----------------------------------
def gera_genoma(): # Retorna um Genoma
    return random.choices([0, 1], k=tam_genoma) # Gerar uma lista de 0 e 1 com o tamanho do Genoma.

def gera_populacao(): # Retorna a população (lista de listas)
    return [gera_genoma() for _ in range(pop_size)]

def print_population(population): # Printa população
    for i in range(pop_size):
        print(f"[Genoma #{i}]: {''.join(str(Gene) for Gene in population[i])}")


#---------------------------------- APTIDAO ----------------------------------
def convert(population): # Reparte e etorna x e y na base 10
    # Divide o genoma em 2
    x = population[:int(tam_genoma/2)]
    y = population[int(tam_genoma/2):]

    # Representação de x, y em string
    strx = [str(x) for x in x]
    stry = [str(y) for y in y]

    # Transforma x, y numa string vazia
    x = y = "" 

    # Transforma x, y em decimal
    x = int(x.join(strx), 2)
    y = int(y.join(stry), 2)

    return x, y

def tratamento(x, y): # Tratamento das variáveis
    x = (x*200/(2**22-1))-100
    y = (y*200/(2**22-1))-100
    return x, y

def fitness_function(x, y): # Retorna o valor de F6 (max = 1)
    aux = x**2 + y**2
    F6 = 0.5 - ((np.sin(np.sqrt(aux))**2-0.5) / (1+(0.001*aux))**2)
    return F6

def calcula_aptidao(population): # Retorna uma lista com as aptidoes
    aptidao = []
    for i in range(pop_size):         
        x, y = convert(population[i])
        x, y = tratamento(x, y)            
        apt = fitness_function(x, y)            
        aptidao.append(apt)
    return aptidao


#---------------------------------- SELECAO ----------------------------------
def selecao_por_roleta(aptidao): # Retorna uma lista com a populacao selecionada
    nova_populacao = [] #Lista vazia da proxima geracao
    
    for _ in range(pop_size):
        soma_parcial = 0
        rand_numb = random.uniform(0, sum(aptidao))
        i = -1
        for aptd in aptidao: 
            i+=1
            soma_parcial += aptd
            if(soma_parcial >= rand_numb):
                nova_populacao.append(Populacao[i])
                break
    return nova_populacao 


#---------------------------------- CROSSOVER ----------------------------------
taxa_crossover = 0.65

def cross(a, b):
    posicao_corte = random.randint(1, tam_genoma-2)
    son1 = a[:posicao_corte] + b[posicao_corte:]
    son2 = b[:posicao_corte] + a[posicao_corte:]
    
    return son1, son2

def gera_cross(population):
    nova_populacao = [] #Lista vazia da proxima geracao
    i = 0
    
    while i < pop_size:
        cromossoma1 = population[i]
        cromossoma2 = population[i+1]
        rand_numb = np.random.uniform()

        if(rand_numb < taxa_crossover): # Efetua cross
            son1, son2 = cross(cromossoma1, cromossoma2)
        else: # Copia genitores
            son1 = cromossoma1
            son2 = cromossoma2
        
        nova_populacao.append(son1)
        nova_populacao.append(son2)
        i+=2

    return nova_populacao


#---------------------------------- MUTAÇÃO ----------------------------------
taxa_mutacao = 0.008

def mutation(population):
    nova_populacao = []

    for i in range(pop_size):
        for gene in population[i]:
            rand_numb = np.random.uniform()
            if rand_numb < taxa_mutacao:
                if gene == 1: gene = 0
                else: gene = 1
        nova_populacao.append(population[i])
    return nova_populacao


#---------------------------------- ELITISMO ----------------------------------
def elitism(parent_population, child_population):
    """
    O operador elitismo consiste em replicar o melhor indivíduo de uma geração,
    inalterado, na geração subsequente
    """
    aptd_pais = calcula_aptidao(parent_population)
    aptd_filhos = calcula_aptidao(child_population)
    
    maior = max(aptd_pais)
    menor = min(aptd_filhos)

    child_population[aptd_filhos.index(menor)] = parent_population[aptd_pais.index(maior)]

    return child_population


#------------------------------ ENCONTRA A SOLUÇÃO ------------------------------
def encontra_solucao(aptidao): # Retorna a aptitude 1, caso encontrada
    for apt in aptidao:
        if(round(apt, 5) == 1): return apt
    return -1

#---------------------------------- MAIN ----------------------------------
# Geração da população
Populacao = gera_populacao()

# Eixo das ordenadas contendo as melhores aptitudes de cada geração
ordn = []

# Algoritmo Genético em ação
for i in range(num_ger):
    print(f"\nGeração #{i}")
    #print_population(Populacao) # Ative se quiser ver as populações a cada geração

    # Calcula aptidao
    aptidao = calcula_aptidao(Populacao)

    # Print da aptidão máxima e de seu dono
    print(f"Aptidão max: {round(max(aptidao), 5)}\nGenoma: {aptidao.index(max(aptidao))}")

    # Adiciona a aptidão máxima no eixo das ordenadas
    ordn.append(round(max(aptidao), 5))

    # Printa Genoma que contem solução
    if encontra_solucao(calcula_aptidao(Populacao)) != -1:
        print(f"SOLUÇÃO:\nGeração: #{i}\nAptidão: {encontra_solucao(calcula_aptidao(Populacao))}")
        break

    # Aplica seleção pela roleta
    Nova_Populacao = selecao_por_roleta(aptidao)

    # Gera populacao de filhos
    Nova_Populacao = gera_cross(Nova_Populacao)

    # Mutação dos filhos
    Nova_Populacao = mutation(Nova_Populacao)

    # O pior morre, o melhor permanece
    Nova_Populacao = elitism(Populacao, Nova_Populacao)

    # Populacao de filhos agora é de pais
    Populacao = Nova_Populacao

#---------------------------------- GRÁFICO ----------------------------------
absc = np.array(range(num_ger)) # Gerações

plt.plot(absc, ordn)
plt.show()
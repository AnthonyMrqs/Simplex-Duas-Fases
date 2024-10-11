import numpy as np

def add_var(n, A, idxs = None): # Função para adicionar as variáveis de folga
    identity = np.identity(n) # Cria a matriz identidade

    if idxs == None:
        idxs = []

    for i in idxs: #Se o parâmtro idx estiver preenchido, quer dizer que teremos desigualdades inversas
        identity[i][i] = -identity[i][i]

    return np.concatenate((A, identity), axis=1)


def delete_duplicate(A): # Função para deleter colunas duplicadas
    At = A.T
    _, b = np.unique(At, axis=0, return_index=True) # Coleta os indices das colunas únicas
    b = np.sort(b) # Organiza o array b para que a ordem da matriz não seja afetada
    Af = At[b] # Matriz com as colunas únicas

    return Af.T
    

def scaling(A, pivo): # Função para escalonar uma matriz
    i, j = pivo # coordenadas do elem pivo
    elem = A[i][j] # elemento pivô

    # Confere se o elemento pivô = 1
    if elem != 1: 
        A[i] = [A[i][k]/elem for k in range(len(A[0]))]
    
    #Escalona a matriz
    for n in range(len(A)):
        if n != i:
            p = A[n][j]
            for m in range(len(A[0])):
                A[n][m] = A[n][m] - p*A[i][m]
    
    return A    


def check_base(m): # Função para explicitar a base no tableau
    identity = np.identity(m.shape[0])
    mb = np.array(m.T[:-1]) # Retira a coluna B e faz a transposição do tableau 
    aux = []

    # Compara cada linha ou coluna da matriz identidade com a matriz mb afim de verificar se existe a matriz identidade está dentro do tableau
    for id_colunm in identity:
        for colunm in mb:
            if np.array_equal(id_colunm, colunm):
                aux.append(id_colunm)
                break

    aux = np.array(aux) # Array com as colunas da matriz identidade encontradas

    not_id_col = [] # Array onde haverá os indíces das colunas que não fazem parte da matriz identidade

    # Compara se as colunas identidade achadas no tableau não pertencem à matriz identidade 
    for i, col in enumerate(mb):
        if not any(np.array_equal(col, col_a) for col_a in aux):
            not_id_col.append(i)
    
    # Se o array estiver vazio, isto significa que que a base está explícita e retorna o tableau sem alterações
    if not not_id_col:
        return m
    
    diference = []

    # Coleta as colunas da matriz identidade que faltam
    for line in identity:
        if not any(np.array_equal(line, aux_line) for aux_line in aux):
            diference.append(line)
    
    # Escalona a matriz de forma que as colunas identidade que faltavam apareçam no tableau
    for i, line in enumerate(diference):
        j = np.where(line == 1)[0][0] 
        pivo = (j, not_id_col[i])

        m = scaling(m, pivo)
    
    return m


def argmin(b, v): # Função para calcular o o argmin do vetor b_i/a_iq
    a_s = np.array(v) # Fazendo uma lista com os elementos que vâo se tornar os denominadores
    if np.all(a_s <= 0): # confere se todos os a's são negativos, se sim, o problema é ilimitado e retorna vazio.
        return None
    
    ps = np.array([b[i]/a_s[i] if a_s[i] > 0  else np.inf for i in range(len(b))]) # Calcula todos os elementos que vão ser calculados no argmin
            
    return np.argmin(ps)


def collect_indexes(t): # Coleta os índices das variáveis básicas e não básicas
    identity = np.identity(len(t)) # Cria uma matriz identidade de acordo com o numero de linhas do tableau inicial
    
    t = t.T[:-1] # Retira a coluna B e faz a transposição do tableau 
    tb = np.array(t) 
    ctb = []

    # Compara cada linha ou coluna da matriz identidade com a matriz tb afim de procurar onde estão os indices da base
    for id_colunm in identity:
        for i, colunm in enumerate(tb):
            if np.array_equal(id_colunm, colunm):
                ctb.append(i)
                break     
    ctn = []
    
    # Coleta todos os indices dos coeficientes que estão fora da base
    ctn = [i for i in range(len(t)) if i not in ctb]

    return ctb, ctn


def two_phase(tableau, b, cfs1): # Função responsável por realizar o método de duas fases
    dim = len(tableau)
    cfs2 = np.append(np.zeros(len(cfs1)), [-1]*dim) # Adiciona yn aos coeficientes e zera os coeficientes anteriores
    
    A = algorithm_simplex(tableau, b, cfs2) 
    return A


def algorithm_simplex(A, b, cfs): # Função responsável por realizar o algoritimo simplex
    A = check_base(A) # Escalona a matriz caso a base esteja implícita
        
    # Realiza um laço infinito até a solução ótima for achada ou o problema for ilimitado
    while True:   
        i_ctb, i_ctn = collect_indexes(A) # Indice das variáveis básicas e não básicas

        An = np.array([A.T[k] for k in i_ctn]).T # Monta a matriz a partir dos indíces de variáveis não básicas

        # Variáveis básicas e não básicas
        ctb = [cfs[i] for i in i_ctb]
        ctn = [cfs[j] for j in i_ctn]

        # Operações para achar o vetor dos coeficientes de custo reduzido
        z = ctb @ An
        r = ctn - z

        # Confere se o vetor de custo está dentro das condições para solução ser ótima
        if not np.all(r == 0) and np.all(r <= 0):
            return A
        
        # Usa a regra proposta por George Dantzig (usar o maior coeficiente de custo) para sabe qual variável q da matriz An irá entrar na base e armazena a coluna q
        q = np.argmax(r)
        vector_q =  An.T[q]

        # Calcula o indíce de variável que irá sair da base
        p = argmin(b, vector_q)      

        # Caso o p seja vazio, o problema é ilimitado
        if p == None:
            print("\nProblema Ilimitado.")
            return None
        
        # Troca a base de acordo com os indíces p e q
        A = scaling(A, (p, i_ctn[q]))


def solutions(A, f): # Função para expor a solução do PPL
    sols = A[:, -1] # Coleta a ultima coluna da matriz (soluções)

    x_values = np.zeros(len(A[0]) - 1) # Inicializa uma matriz de 0's do tamanho das soluções 

    #Percorre as colunas da matriz 
    for i in range(A.shape[1] - 1):
        col = A[:, i] 
        if np.count_nonzero(col) == 1 and 1 in col: # Confere se a coluna faz parte da matriz identidade
            indice = np.where(col == 1)[0][0] # Coleta o indíce onde se encontra o 1 na coluna
            x_values[i] = sols[indice]

    x_values = x_values[:len(f)] # Ignora as variáveis de folga e coleta apenas as soluções das variáveis da função objetivo
    result = sum(x_values[i]*e for i, e in enumerate(f)) # Calcula a solução ótima
    
    return x_values, result


def simplex(A, b, c, eq = None): # Função que realiza a preparação do tableau para as operações do algoritimo
    n = len(A)
    idx = [i for i in range(len(b)) if b[i] < 0] # Lista de índices que tem a desigualdade >=
    b = np.abs(b) # Aplicao o valor absoluto em todos os elementos da coluna b

    # Confere se o problema se trata de igualdade, desigualdades inversas ou desigualdades iguais
    if eq:
        tableau = A
    elif len(idx) > 0:
        tableau = add_var(n, A, idx) # Adiciona as variaveis de folga
        tableau = add_var(n, tableau) # Adiciona as variáveis artificiais
    else:
        tableau = add_var(n, A)


    tableau = delete_duplicate(tableau) # Retira as colunas duplicadas
    tableau = np.hstack((tableau, b.reshape(-1, 1))) # Concatena a coluna B ao tableau

    coeficientes = np.append(c, [0]*n) # Adicionando os coeficentes das variaveis de folga

    # Caso o problema apresente desigualdade inversas, será realizado o simplex de duas fases
    if idx:
        tableau = two_phase(tableau, b, coeficientes) # Tableau final do método duas fases

        if np.any(tableau == None):
            return 
        
        tableau = np.concatenate((tableau[:, :coeficientes.shape[0]], tableau[:, -1:]), axis=1) # Retira as colunas y's do tableau

    
    # Realiza o simplex para resolver o PPL
    result = algorithm_simplex(tableau, b, coeficientes)
    
    # Confere se o programa achou a solução e expõe a solução
    if np.any(result != None):
        return solutions(result, c)


c = np.array([])
A = np.array([]) 

b = np.array([])

res = simplex(A, b, c) # Tupla de valores (valores de x, aplicação na função objetivo)

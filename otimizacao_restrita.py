# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import numdifftools as nd
import time


def processarFuncao(funcao_input, suprimir_aviso=False, simbolos=None):
    try:
        # caso a entrada seja string, criaremos funções analíticas
        expr = sp.sympify(funcao_input)
        variaveis = list(expr.free_symbols)
        funcao_lambdify = sp.lambdify(variaveis, expr)
        if not suprimir_aviso: print('Ordem das variáveis: {}'.format(variaveis))
        if simbolos is not None: [simbolos.append(variavel) for variavel in variaveis]

        gradiente_simbolico = [expr.diff(variavel) for variavel in variaveis]
        hessiana_simbolica = [[derivada_parcial.diff(variavel) for variavel in variaveis] for derivada_parcial in
                              gradiente_simbolico]

        gradiente_lambda = [sp.lambdify(variaveis, derivada_parcial) for derivada_parcial in gradiente_simbolico]
        hessiana_lambda = [[sp.lambdify(variaveis, derivada_parcial) for derivada_parcial in gradiente] for gradiente in
                           hessiana_simbolica]

        funcao = lambda x: funcao_lambdify(*x)
        gradiente = lambda x: np.array(list(map(lambda g: g(*x), gradiente_lambda)))
        hessiana = lambda x: np.array(list(map(lambda grad: list(map(lambda g: g(*x), grad)), hessiana_lambda)))

    except sp.SympifyError:
        # caso a entrada seja uma função do Python, vamos utilizar diferenças finitas
        funcao = funcao_input
        gradiente = nd.Gradient(funcao)
        hessiana = nd.Hessian(funcao)

    return funcao, gradiente, hessiana

def plotar_teste_curvas_de_nivel(x_final, F, trajetoria, limx, limy, plotar_trajetoria=True, levels=50, eixos=None):
    xaxis = np.linspace(*limx, 100)
    yaxis = np.linspace(*limy, 100)
    X, Y = np.meshgrid(xaxis, yaxis)
    Z = F([X, Y])
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=levels)
    ax.plot(x_final[0], x_final[1], 'ro')
    ax.set_title('Número de saltos: %d' % (len(trajetoria) - 1), color='magenta')
    ax.tick_params(colors='magenta')
    if eixos is not None:
        ax.set_xlabel(eixos[0], color='magenta')
        ax.set_ylabel(eixos[1], color='magenta', rotation='horizontal', horizontalalignment='right')

    if plotar_trajetoria:
        for i in range(len(trajetoria) - 1):
            ax.plot(*list(zip(trajetoria[i], trajetoria[i + 1])), 'r')
            ax.set_xlim(limx)
            ax.set_ylim(limy)


def passo_metodo_secao_aurea(phi, x, d, epsilon=1e-6, rho=1):
    theta1 = (3 - 5 ** (1 / 2)) / 2
    theta2 = 1 - theta1
    # fase 1 - obtenção do intervalo [a, b]
    (a, s, b) = (0, rho, 2 * rho)
    while phi(x, b, d) < phi(x, s, d):
        a, s, b = s, b, 2 * b
    # fase 2 - obtenção de t_barra em [a, b]
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    while b - a > epsilon:
        if phi(x, u, d) < phi(x, v, d):
            b = v
            v = u
            u = a + theta1 * (b - a)
        else:
            a = u
            u = v
            v = a + theta2 * (b - a)
    return (u + v) / 2


def D(phi, x, d, delta=1e-6):  # derivada primeira
    return lambda x, t, d: (phi(x, t + delta, d) - phi(x, t, d)) / delta


def D2(phi, x, d, delta=1e-6):  # derivada segunda
    return lambda x, t, d: (phi(x, t + 2 * delta, d) - 2 * phi(x, t, d) + phi(x, t - 2 * delta, d)) / (4 * delta ** 2)


from math import inf


def passo_metodo_newton(phi, x, d, t_inicial, erro=1e-6, precisao=1e-6):
    t_anterior = -inf
    t = t_inicial
    d_phi_dt = D(phi, x, d, precisao)  # d_phi_dt(t) retorna uma aproximação para a derivada de phi
    d2_phi_dt2 = D2(phi, x, d, precisao)  # d2_phi_dt2(t) retorna uma aproximação para a derivada segunda de phi
    while abs(t - t_anterior) > erro:
        t_anterior = t
        if d2_phi_dt2(x, t_anterior, d) == 0:
            raise Exception('Divide by zero')
        t = t_anterior - d_phi_dt(x, t_anterior, d) / d2_phi_dt2(x, t_anterior, d)
    return t


def passo_metodo_armijo(funcao, gradiente, x, direcao, eta=1 / 2):
    t = 1
    while True:
        delta_linear = eta * t * np.dot(gradiente(x), direcao)
        salto = funcao(x + t * direcao)
        if salto <= funcao(x) + delta_linear:
            break
        t *= 0.8
    return t


def gerar_funcao_barreira(restricoes, tipo_de_barreira):
    if tipo_de_barreira == "log":
        return lambda x: -sum([np.log(-restricao) for restricao in restricoes(x)])
    elif tipo_de_barreira == "inverso":
        return lambda x: -sum([-1 / restricao for restricao in restricoes(x)])

def minimizador_metodo_newton(funcao_str, x_inicial, direcao=None, erro=1e-6, sigma=0,
                              metodo_de_busca='newton', precisao=1e-6, rho=1, eta=0.5,
                              **kwargs):
    # sanitizando a entrada do usuário para compatibilidade com o Numpy
    x_inicial = np.array(x_inicial, dtype='float64')
    # processando o argumento funcao_str
    funcao, gradiente, hessiana = processarFuncao(funcao_str, suprimir_aviso=kwargs[
        'suprimir_aviso'] if 'suprimir_aviso' in kwargs else False)
    # determinando a dimensão do domínio
    n = len(gradiente(x_inicial))

    # determinando a direção de descida
    if direcao is not None:
        direcao = np.array(direcao)
    else:
        direcao = -gradiente(x_inicial)

    # iniciando algoritmo de otimização
    x_final = x_inicial
    trajetoria = [list(x_final)]
    phi = lambda x, t, d: funcao(x + t * d)
    t_inicial = kwargs['t_inicial'] if 't_inicial' in kwargs else 1
    while np.linalg.norm(direcao) > erro:
        if metodo_de_busca == 'secao_aurea':
            t_k = passo_metodo_secao_aurea(phi, x_final, direcao, erro, rho)
        elif metodo_de_busca == 'newton':
            try:
                t_k = passo_metodo_newton(phi, x_final, direcao, t_inicial, erro, precisao)
            except:  # neste caso, também atingimos o limite de precisao e devemos parar
                break
        elif metodo_de_busca == 'armijo':
            t_k = passo_metodo_armijo(funcao, gradiente, x_final, direcao, eta)
        else:
            print('Método de busca do passo inválido.')
            break
        # calculando o fator M: inverso da hessiana modificada (pelo parâmetro sigma)
        M_k = np.linalg.inv(hessiana(x_final) + sigma * np.eye(n))
        # cálculo da direção no método de Newton
        direcao = - M_k @ gradiente(x_final)
        # atualização do passo
        x_final += t_k * direcao
        # catalogando o passo
        trajetoria.append(list(x_final))
        # podemos atingir um ponto estacionário antes de obtermos a precisão desejada
        if (len(trajetoria) > 1 and trajetoria[-1] == trajetoria[-2]):
            print('Aviso: ponto estacionário atingido antes do critério de precisão.')
            break

    valor_final = funcao(x_final)

    return x_final, valor_final, trajetoria

def minimizador_metodo_gradientes_conjugados(funcao_input, restricoes, x_inicial, direcao=None,
                                             erro=1e-6, modalidade='classico',
                                             metodo_de_busca='newton', precisao=1e-6,
                                             rho=1, eta=0.5, **kwargs):
    # sanitizando a entrada do usuário para compatibilidade com o Numpy
    x_inicial = np.array(x_inicial, dtype='float64')
    # processando o argumento funcao
    funcao, gradiente, hessiana = processarFuncao(funcao_input, suprimir_aviso=kwargs['suprimir_aviso'] if 'suprimir_aviso' in kwargs else False)
    barreira = gerar_funcao_barreira(restricoes, 'log')
    # determinando a direção de descida
    if direcao is not None:
        direcao = np.array(direcao)
    else:
        direcao = -gradiente(x_inicial)
    # iniciando algoritmo de otimização
    x_final = x_inicial
    trajetoria = [list(x_final)]
    phi = lambda x, t, d: funcao(x + t * d)
    t_inicial = kwargs['t_inicial'] if 't_inicial' in kwargs else 1
    while np.linalg.norm(gradiente(x_final)) > erro:
        if metodo_de_busca == 'secao_aurea':
            t_k = passo_metodo_secao_aurea(phi, x_final, direcao, erro, rho)
        elif metodo_de_busca == 'newton':
            try:
                t_k = passo_metodo_newton(phi, x_final, direcao, t_inicial, erro, precisao)
            except:  # neste caso, também atingimos o limite de precisao e devemos parar
                break
        elif metodo_de_busca == 'armijo':
            t_k = passo_metodo_armijo(funcao, gradiente, x_final, direcao, eta)
        elif metodo_de_busca == 'quadratico':
            t_k = - gradiente(x_final) @ direcao / (direcao @ hessiana(x_final) @ direcao)
        else:
            print('Método de busca do passo inválido.')
            break
        x_anterior = np.copy(x_final)
        x_final += t_k * direcao
        while len(restricoes(x_final)) > 0 and not max(restricoes(x_final)) <= 0:
            t_k *= 0.9
            x_final = x_anterior + t_k * direcao
            if np.linalg.norm(x_final - x_anterior) <= erro:
                x_final = x_anterior
                trajetoria.append(list(x_final))
                return
        if modalidade == 'classico':
            beta_k = direcao @ hessiana(x_final) @ gradiente(x_final) / (direcao @ hessiana(x_final) @ direcao)
        elif modalidade == 'fletcher-reeves':
            beta_k = gradiente(x_final) @ gradiente(x_final) / (gradiente(x_anterior) @ gradiente(x_anterior))
        else:
            print('Valor inválido para o parâmetro: modalidade.')
            break
        direcao = -gradiente(x_final) + beta_k * direcao
        trajetoria.append(list(x_final))
        
        print("{}. x: {}, L(x, mu): {} \n    B(x): {}, mu: {}, grad(x): {}".format(len(trajetoria), x_final, funcao(x_final), barreira(x_final), kwargs['mu'], gradiente(x_final)))
        
        # podemos atingir um ponto estacionário antes de obtermos a precisão desejada
        if len(trajetoria) > 1 and trajetoria[-1] == trajetoria[-2]:
            print('Aviso: ponto estacionário atingido antes do critério de precisão.')
            break
    valor_final = funcao(x_final)
    return x_final, valor_final, trajetoria


# função utilidade para transpor um vetor linha em vetor coluna
transpor_vetor = lambda v: np.array([[v[i]] for i in range(len(v))])


def minimizador_metodo_quase_newton_rank_1(funcao_str, x_inicial, direcao=None,
                                           erro=1e-6, beta=1,
                                           estabilizar=False,
                                           metodo_de_busca='newton',
                                           precisao=1e-6, rho=1, eta=0.5, **kwargs):
    # sanitizando a entrada do usuário para compatibilidade com o Numpy
    x_inicial = np.array(x_inicial, dtype='float64')
    # processando o argumento funcao_str
    funcao, gradiente, _ = processarFuncao(funcao_str, suprimir_aviso=kwargs[
        'suprimir_aviso'] if 'suprimir_aviso' in kwargs else False)
    # determinando a dimensao do domínio
    n = len(gradiente(x_inicial))

    # determinando a direção de descida
    if direcao is not None:
        direcao = np.array(direcao)
    else:
        direcao = -gradiente(x_inicial)

    # iniciando algoritmo de otimização
    x_final = np.copy(x_inicial)
    trajetoria = [list(x_final)]
    phi = lambda x, t, d: funcao(x + t * d)
    # Definindo valor inicial da aproximação da Hessiana
    H = 1 / beta * np.eye(n)
    t_inicial = kwargs['t_inicial'] if 't_inicial' in kwargs else 1
    while np.linalg.norm(direcao) > erro:
        if metodo_de_busca == 'secao_aurea':
            t = passo_metodo_secao_aurea(phi, x_final, direcao, erro, rho)
        elif metodo_de_busca == 'newton':
            try:
                t = passo_metodo_newton(phi, x_final, direcao, t_inicial, erro, precisao)
            except:  # neste caso, também atingimos o limite de precisao e devemos parar
                break
        elif metodo_de_busca == 'armijo':
            t = passo_metodo_armijo(funcao, gradiente, x_final, direcao, eta)
        else:
            print('Método de busca do passo inválido.')
            break
        # cálculo da direção no método de Newton
        direcao = - H @ gradiente(x_final)
        # atualização do passo
        x_final += t * direcao
        # parâmetros de estimação da Hessiana
        p = t * direcao
        q = gradiente(x_final) - gradiente(x_final - p)
        # critérios de estabilização de H (denominador -> 0 ou hessiana não definida positiva)
        if estabilizar and (
                abs(q @ (p - H @ q)) < 1e-8 * np.linalg.norm(q) * np.linalg.norm(p - H @ q) or p @ q - q @ H @ q < 0):
            # estabilizando H
            mu = (p @ p) / (q @ p) - np.sqrt(((p @ p) ** 2) / ((q @ p) ** 2) - (p @ p) / (q @ q))
            H = np.eye(n) * mu
        else:
            H += ((p - H @ q) * transpor_vetor(p - H @ q)) / ((p - H @ q) @ q)
        # catalogando o passo
        trajetoria.append(list(x_final))
        # podemos atingir um ponto estacionário antes de obtermos a precisão desejada
        if len(trajetoria) > 1 and trajetoria[-1] == trajetoria[-2]:
            print('Aviso: ponto estacionário atingido antes do critério de precisão.')
            break

    valor_final = funcao(x_final)

    return x_final, valor_final, trajetoria


    
def processarFuncaoLagrangiana(funcao_input, restricoes, mu, tipo_de_barreira, suprimir_aviso=False, simbolos=None):
    try:
        # caso a entrada seja string, criaremos funções analíticas
        expr = sp.sympify(funcao_input)
        variaveis = list(expr.free_symbols)
        if not suprimir_aviso: print('Ordem das variáveis: {}'.format(variaveis))
        if simbolos is not None: [simbolos.append(variavel) for variavel in variaveis]
        
        restricoes_simbolicas = [sp.sympify(restricao) for restricao in restricoes]
        
        if tipo_de_barreira == "log":
            funcao_barreira_simbolica = -sum([sp.log(-restricao) for restricao in restricoes_simbolicas])
        elif tipo_de_barreira == "inverso":
            funcao_barreira_simbolica = -sum([-1 / restricao for restricao in restricoes_simbolicas])
            
        lagrangiana = str(expr + mu * funcao_barreira_simbolica)
        
        restricoes_lambda = [sp.lambdify(variaveis, restricao) for restricao in restricoes_simbolicas]
        funcao_barreira_lambda = sp.lambdify(variaveis, funcao_barreira_simbolica)
        
        restricoes = lambda x: np.array(list(map(lambda g: g(*x), restricoes_lambda)))
        funcao_barreira = lambda x: funcao_barreira_lambda(*x)
        
        return lagrangiana, restricoes, funcao_barreira
        
    except sp.SympifyError:
        # caso a entrada seja uma função do Python, vamos utilizar diferenças finitas
        funcao_barreira = gerar_funcao_barreira(restricoes, tipo_de_barreira)
        def lagrangiana(x): funcao_input(x) + mu * funcao_barreira(x)
        return lagrangiana, restricoes, funcao_barreira

def minimizador_metodo_barreira(funcao_input, restricoes_input, x_inicial, direcao=None,
                                metodo_de_minimizacao="gradientes-conjugados",
                                metodo_de_busca="newton", tipo_de_barreira="log",
                                mu=10, beta=0.1, erro=1e-6, precisao=1e-6,
                                rho=1, eta=0.5):
    x_anterior = np.copy(x_inicial)
    trajetoria = []
    while True:
        lagrangiana, restricoes, funcao_barreira = processarFuncaoLagrangiana(funcao_input, restricoes_input, mu, tipo_de_barreira, suprimir_aviso=True)
        if metodo_de_minimizacao == "gradientes-conjugados":
            x_final, valor_final, trajetoria_parcial = minimizador_metodo_gradientes_conjugados(lagrangiana, restricoes,
                                                                                                x_anterior, direcao,
                                                                                                erro, "classico",
                                                                                                metodo_de_busca, precisao,
                                                                                                rho, eta, suprimir_aviso=True, mu=mu)
        trajetoria += trajetoria_parcial
        if abs(mu * funcao_barreira(x_final)) < erro:
            break
        mu *= beta
        x_anterior = x_final
    return x_final, valor_final, trajetoria

def executar_teste (funcao_input, restricoes_input, x_inicial, metodo_de_minimizacao="gradientes-conjugados",
                    metodo_de_busca="newton", tipo_de_barreira="log", mu=10, beta = 0.1, erro=1e-6,
                    precisao=1e-6, rho=1, eta=0.5):
    x_final, valor_final, trajetoria = minimizador_metodo_barreira(funcao_input, restricoes_input, x_inicial, None, metodo_de_minimizacao,
                                                                   metodo_de_busca, tipo_de_barreira, mu, beta, erro, precisao, rho, eta)
    output_str = "x_final: {}, valor_final: {}\n".format(x_final, valor_final)
    output_str += "número de passos: {}\n".format(len(trajetoria) - 1)
    print(output_str)
"""
@file gradient_descent.py
@brief Implementa o algoritmo de descida do gradiente para regressão linear.
"""

import numpy as np
from Functions.compute_cost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    @brief Executa a descida do gradiente para minimizar a função de custo J(θ)
           no contexto de regressão linear.

    A cada iteração, os parâmetros theta são atualizados com base
    no gradiente da função de custo em relação aos parâmetros atuais.

    @param X np.ndarray
        Matriz de entrada (m amostras × n atributos), incluindo termo de bias.

    @param y np.ndarray
        Vetor de saída esperado com dimensão (m,).

    @param theta np.ndarray
        Vetor de parâmetros inicial (n,).

    @param alpha float
        Taxa de aprendizado (learning rate).

    @param num_iters int
        Número de iterações da descida do gradiente.

    @return tuple[np.ndarray, np.ndarray, np.ndarray]
        theta: vetor otimizado de parâmetros (n,).
        J_history: vetor com o histórico do valor da função de custo em cada iteração (num_iters,).
        theta_history: parâmetros em cada iteração (num_iters+1, n).
    """

    m = len(y)
    J_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters + 1, theta.shape[0]))
    theta_history[0] = theta

    for i in range(num_iters):
        predictions = X @ theta
        erro = predictions -y
        gradient = (1 / m) * (X.T @ erro)
        theta = theta - (alpha * gradient)


        J_history[i] = compute_cost(X, y, theta)
        theta_history[i + 1] = theta

    return theta, J_history, theta_history

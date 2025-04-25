"""
@file compute_cost.py
@brief Calcula o custo para regressão linear.
"""

import numpy as np

def compute_cost(X, y, theta):
    """
    @brief Calcula o custo para regressão linear.

    Esta função calcula o valor da função de custo de regressão linear (erro quadrático médio):
    J(θ) = (1 / (2 * m)) * Σ (h(θ) - y)^2

    onde:
    - J(θ) é o custo
    - m é o número de exemplos de treinamento
    - h(θ) é a função hipótese (X @ theta)
    - y é o vetor de valores observados

    @param X np.ndarray
        Matriz de atributos com termo de intercepto incluso (shape: m x n).

    @param y np.ndarray
        Vetor de variáveis alvo (shape: m,).

    @param theta np.ndarray
        Vetor de parâmetros da regressão linear (shape: n,).

    @return float
        Valor do custo computado.
    """

    m = len(X)
    h_o = X @ theta
    errors = h_o - y
    J_o = (1 / (2 * m)) * np.sum(errors ** 2)

    return J_o
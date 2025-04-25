"""
@file warm_up_exercise.py
@brief Conjunto de funções introdutórias para regressão linear com NumPy.
"""

import numpy as np


def warm_up_exercise1():
    """
    @brief Cria e retorna uma matriz identidade 5x5.

    @return np.ndarray
        Matriz identidade de dimensão 5x5.
    """
    return np.identity(5)


def warm_up_exercise2(m=5):
    """
    @brief Cria um vetor coluna de 1s, usado como termo de bias em regressão linear.

    @param m int
        Número de exemplos (linhas).

    @return np.ndarray
        Vetor de shape (m, 1) com todos os valores iguais a 1.
    """
    return np.ones((m, 1))


def warm_up_exercise3(x):
    """
    @brief Adiciona uma coluna de 1s (bias) ao vetor de entrada x.

    @param x np.ndarray
        Vetor unidimensional de shape (m,).

    @return np.ndarray
        Matriz de shape (m, 2), com a primeira coluna sendo 1s (bias) e a segunda os valores de x.
    """
    m = len(x)
    x = np.reshape(x, (m, 1))
    bias = np.ones((m, 1))
    return np.hstack((bias, x))


def warm_up_exercise4(X, theta):
    """
    @brief Realiza a multiplicação matricial entre X e θ, simulando h(θ) = X @ θ.

    @param X np.ndarray
        Matriz de entrada de shape (m, n).

    @param theta np.ndarray
        Vetor de parâmetros de shape (n,).

    @return np.ndarray
        Vetor de predições de shape (m,).
    """
    return np.dot(X, theta)


def warm_up_exercise5(predictions, y):
    """
    @brief Calcula o vetor de erros quadráticos entre as predições e os valores reais.

    @param predictions np.ndarray
        Vetor de predições (m,).

    @param y np.ndarray
        Vetor de valores reais (m,).

    @return np.ndarray
        Vetor com os erros quadráticos: (pred - y)^2.
    """
    return (predictions - y) ** 2


def warm_up_exercise6(errors):
    """
    @brief Calcula o custo médio (mean cost) a partir dos erros quadráticos.

    @param errors np.ndarray
        Vetor de erros quadráticos (m,).

    @return float
        Custo médio (mean cost).
    """
    return np.mean(errors) / 2


def warm_up_exercise7(X, y, theta):
    """
    @brief Calcula o custo médio (mean cost) para um modelo de regressão linear.

    @param X np.ndarray
        Matriz de entrada de shape (m, n).

    @param y np.ndarray
        Vetor de valores reais (m,).

    @param theta np.ndarray
        Vetor de parâmetros de shape (n,).

    @return float
        Custo médio (mean cost).
    """
    predictions = warm_up_exercise4(X, theta)
    errors = warm_up_exercise5(predictions, y)
    mean_cost = warm_up_exercise6(errors)
    return mean_cost

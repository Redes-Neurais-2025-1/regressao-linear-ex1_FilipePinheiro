"""
@file plot_data.py
@brief Plota os pontos de dados do conjunto de treinamento.
"""

import matplotlib.pyplot as plt


def plot_data(x, y):
    """
    @brief Plota os dados de treinamento como cruzes vermelhas.

    @param x np.ndarray
        Variável independente (população).

    @param y np.ndarray
        Variável dependente (lucro).
    """
    plt.figure()
    plt.plot(x, y, 'rx', markersize=5)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Training Data')
    plt.grid(True)
    plt.show()

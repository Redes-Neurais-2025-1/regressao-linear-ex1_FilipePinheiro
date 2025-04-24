import numpy as np

def mean_squared_error_vectorized(x, y, theta):
    """
    Calcula o Mean Squared Error de forma vetorial.
    
    Parâmetros:
    x     : array de entrada com bias (ex: [1, x1])
    y     : array de saídas reais
    theta : array de parâmetros do modelo
    
    Retorna:
    mse   : erro quadrático médio (MSE)
    """
    m = len(y)
    predictions = x @ theta
    error = predictions - y
    mse = (1 / (2 * m)) * np.sum(error ** 2)
    return mse

def mean_squared_error_loop(x, y, theta):
    """
    Calcula o Mean Squared Error utilizando loop.
    
    Parâmetros:
    x     : array de entrada com bias (ex: [1, x1])
    y     : array de saídas reais
    theta : array de parâmetros do modelo
    
    Retorna:
    mse   : erro quadrático médio (MSE)
    """
    m = len(y)
    error_sum = 0
    for i in range(m):
        prediction = np.dot(x[i], theta)
        error_sum += (prediction - y[i]) ** 2
    mse = error_sum / (2 * m)
    return mse

# Exemplo de entrada
x = np.array([[1], [2], [3]])  # Feature original (sem bias)
x_aug = np.hstack([np.ones((3, 1)), x])  # Adicionando coluna de bias
y = np.array([1, 2, 3])
theta = np.array([-1, 2])  # Parâmetros do modelo

# Cálculo
mse_v = mean_squared_error_vectorized(x_aug, y, theta)
mse_l = mean_squared_error_loop(x_aug, y, theta)

print("MSE (vetorial):", mse_v)
print("MSE (loop):    ", mse_l)

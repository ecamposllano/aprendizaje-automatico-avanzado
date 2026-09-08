import numpy as np

def inicializar_pesos(V, N, seed=42):
    """
    |V|: , cantidad de palabras del vocabulario
    N: , dimension de la capa oculta
    """
    generador_num_aleatorios = np.random.default_rng(seed)
    W = generador_num_aleatorios.normal(0, 0.1, size=(V, N))
    W_prima = generador_num_aleatorios.normal(0, 0.1, size=(N, V))
    return W, W_prima


def softmax(u):
    exp_u = np.exp(u - np.max(u))
    return exp_u / np.sum(exp_u)



def propagacion(W, W_prima, c_indices):
    """
    c_indices: lista de C indices de palabras de contexto.
    Como cada x_k es one-hot, W^T @ x_k es simplemente la fila de W
    correspondiente a esa palabra (no armamos vectores one-hot).

    Devuelve h (N,), y (V,), u (V,)
    """
    N = W.shape[1]
    C = len(c_indices)

    h = np.zeros(N)
    for idx in c_indices:
        h += W[idx]
    h = h / C

    u = W_prima.T @ h  # (V,)
    y = softmax(u)

    return h, y, u


def retropropagacion(W, W_prima, c_indices, indice_objetivo, h, y, eta):

    C = len(c_indices)

    # Error de salida: e = y - t  (t es one-hot de la palabra objetivo)
    e = y.copy()
    e[indice_objetivo] = e[indice_objetivo] - 1.0  # (V,)

    # Gradiente de W': dW' = h @ e^T  -> (N, V)
    dW_prima = np.outer(h, e)

    # Error propagado hacia la capa oculta: EH = W' @ e  -> (N,)
    EH = W_prima @ e

    # Actualizo W' completa
    W_prima = W_prima - (eta * dW_prima)

    # Actualizo W solo en las filas de las palabras de contexto
    # (porque x_k es one-hot, el gradiente solo "toca" esas filas)
    for idx in c_indices:
        W[idx] = W[idx] - (eta * (EH / C))

    return W, W_prima
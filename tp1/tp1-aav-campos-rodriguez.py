import numpy as np
import matplotlib.pyplot as plt

def cargaProcesarTexto(ruta_archivo):
    # Leemos el archivo 
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        texto = archivo.read()

    texto = texto.lower()
    tokens = texto.split()

    return tokens


def armarDiccionario(tokens):
    # Palabras únicas ordenadas
    palabras_unicas = sorted(list(set(tokens)))
    V = len(palabras_unicas)

    palabra_a_indice = {}
    indice_a_palabra = {}

    numero = 0
    for palabra in palabras_unicas:
        palabra_a_indice[palabra] = numero
        indice_a_palabra[numero] = palabra
        numero = numero + 1

    return palabra_a_indice, indice_a_palabra, V



def generaContextos(tokens, palabra_a_indice, ventana=4):
    """ventana=4 significa: 4 a la izquierda y 4 a la derecha (C = 8 palabras)

    ventana=5 significa: 5 a la izquierda y 5 a la derecha (C = 10 palabras)
    """
    ejemplos = []
    total_tokens = len(tokens)

    # Recorremos asegurando que haya suficiente margen a izquierda y derecha
    for i in range(ventana, total_tokens - ventana):
        # La palabra del medio es el objetivo
        palabra_objetivo = tokens[i]                                # la palabra en la posición i del texto
        indice_objetivo = palabra_a_indice[palabra_objetivo]        # su índice fijo en el vocabulario

        # Las palabras de alrededor son el contexto
        palabras_izq = tokens[i - ventana : i]
        palabras_der = tokens[i + 1 : i + ventana + 1]
        contexto = palabras_izq + palabras_der

        # Convertimos las palabras del contexto a sus números (índices)
        c_indices = []
        for palabra in contexto:
            c_indices.append(palabra_a_indice[palabra])     # cada palabra del contexto -> su índice   

        ejemplos.append((c_indices, indice_objetivo))   # (indices de las palabras del contexto, inidice de la palabra objetivo)
    return ejemplos     # lista de tuplas (c_indices, indice_objetivo), una por cada palabra central posible


def inicializarPesos(V, N, seed=42):
    """
    |V|: , cantidad de palabras del vocabulario
    N: , dimension de la capa oculta
    """
    np.random.seed(seed)
    W = 0.1 * np.random.randn(V, N)
    W_prima = 0.1 * np.random.randn(N, V)

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

    # Error de salida: e = y - t
    e = y.copy()
    e[indice_objetivo] = e[indice_objetivo] - 1.0  # le resto 1 solo a la posición correcta

    # Gradiente de W': dW' = h @ e^T  -> (N, V)
    dW_prima = np.outer(h, e)

    # Error propagado hacia la capa oculta: EH = W' @ e  -> (N,)
    EH = W_prima @ e

    # Actualizo W' completa
    W_prima = W_prima - (eta * dW_prima)

    # Actualizo W solo en las filas de las palabras de contexto (el gradiente solo "toca" esas filas)
    for idx in c_indices:
        W[idx] = W[idx] - (eta * (EH / C))

    return W, W_prima


def calcularPerdida(y, indice_objetivo):
    """Calcula la función de costo E = -log(y_j*)"""
    return -np.log(y[indice_objetivo])



def similitudCoseno(v_A, v_B):
    """Calcula la similitud del coseno entre dos representaciones vectoriales:

    cos = (v_A . v_B) / (||v_A|| * ||v_B||)

    v_A, v_B: vectores de tamaño (N,) (filas de la matriz W)
    Retorna un valor entre -1 y 1
    """
    norma_A = np.linalg.norm(v_A)
    norma_B = np.linalg.norm(v_B)

    if norma_A == 0 or norma_B == 0:
        return 0.0

    return np.dot(v_A, v_B) / (norma_A * norma_B)


def mostrarPalabrasSimilaresCoseno(W, palabra_buscada, palabra_a_indice, cantidad=5):
    # 1. Buscamos el vector de la palabra que queremos consultar
    indice_buscado = palabra_a_indice[palabra_buscada]
    vector_buscado = W[indice_buscado]

    # 2. Comparamos contra todas las demás palabras del vocabulario
    puntajes = []
    for otra_palabra, otro_indice in palabra_a_indice.items():
        if otra_palabra != palabra_buscada:  # Para no compararla consigo misma
            vector_otro = W[otro_indice]
            similitud = similitudCoseno(vector_buscado, vector_otro)
            # Guardamos primero la similitud para que Python ordene fácil
            puntajes.append((similitud, otra_palabra))

    # 3. Ordenamos de mayor a menor y cortamos las primeras para mostrar solo las más similares
    puntajes.sort(reverse=True)
    mejores = puntajes[:cantidad]

    # 4. Mostramos los resultados
    print(f"\nPalabras más parecidas a '{palabra_buscada}':")
    for similitud, palabra in mejores:
        print(f"  - {palabra}: {similitud:.4f}")


def mostrarPalabrasSimilaresProdVectorial(W, palabra_buscada, palabra_a_indice, cantidad=5):
    # 1. Buscamos el vector de la palabra que queremos consultar
    indice_buscado = palabra_a_indice[palabra_buscada]
    vector_buscado = W[indice_buscado]

    # 2. Comparamos contra todas las demás palabras del vocabulario
    puntajes = []
    for otra_palabra, otro_indice in palabra_a_indice.items():
        if otra_palabra != palabra_buscada:  # Para no compararla consigo misma
            vector_otro = W[otro_indice]
            similitud = vector_buscado @ vector_otro
            # Guardamos primero la similitud para que Python ordene fácil
            puntajes.append((similitud, otra_palabra))

    # 3. Ordenamos de mayor a menor y cortamos las primeras
    puntajes.sort(reverse=True)
    mejores = puntajes[:cantidad]

    # 4. Mostramos los resultados
    print(f"\nPalabras más parecidas a '{palabra_buscada}':")
    for similitud, palabra in mejores:
        print(f"  - {palabra}: {similitud:.4f}")



import time

def entrenar(tokens, ventana, epocas, N, eta):
    palabra_a_indice, indice_a_palabra, V = armarDiccionario(tokens)
    ejemplos = generaContextos(tokens, palabra_a_indice, ventana=ventana)

    print(f" ENTRENANDO CON VENTANA = {ventana} (Contexto de {ventana*2} palabras)")
    print(f" Tokens: {len(tokens)} | Vocabulario |V|: {V} | Ejemplos: {len(ejemplos)}")

    W, W_prima = inicializarPesos(V, N)

    for epoca in range(1, epocas + 1):
        perdida_total = 0.0
        inicio_epoca = time.time()

        for i, (c_indices, indice_objetivo) in enumerate(ejemplos):
            h, y, u = propagacion(W, W_prima, c_indices)
            perdida = calcularPerdida(y, indice_objetivo)
            perdida_total += perdida
            W, W_prima = retropropagacion(W, W_prima, c_indices, indice_objetivo, h, y, eta)

            if i > 0 and i % 2000 == 0:
                transcurrido = time.time() - inicio_epoca
                velocidad = i / transcurrido  # ejemplos por segundo
                restante = (len(ejemplos) - i) / velocidad
                print(f"  ...ejemplo {i}/{len(ejemplos)} | {velocidad:.0f} ej/seg | faltan ~{restante:.0f} seg de esta época")

        perdida_promedio = perdida_total / len(ejemplos)
        duracion_epoca = time.time() - inicio_epoca
        print(f"Época {epoca:02d}/{epocas} - Pérdida: {perdida_promedio:.4f} - Duración: {duracion_epoca:.1f} seg")

    return W, palabra_a_indice, indice_a_palabra



'''  Ernes esto lo dejo comentado para ver como se imprime
tokens = cargaProcesarTexto("tp1/texto_corto.txt")

# Experimento con 4 palabras a cada lado (Ventana = 4)
W_ventana4, palabra_a_idx_4, idx_a_palabra_4 = entrenar(
        tokens, ventana=4, epocas=15, N=50, eta=0.05
    )

    # Experimento con 5 palabras a cada lado (Ventana = 5)
W_ventana5, palabra_a_idx_5, idx_a_palabra_5 = entrenar(
        tokens, ventana=5, epocas=15, N=50, eta=0.05
    )

    # 4. Palabras del texto para examinar la similitud
## CAMBIAR: En vez de escribir palabras a mano que quizás no existen,
# le pedimos a Python que elija 5 palabras AL AZAR del vocabulario, con random y seed.

palabras_para_probar = ["caballo", "fábulas", "mundo"]

print("Resultado con ventana = 4")
for palabra in palabras_para_probar:
        mostrarPalabrasSimilares(
            W_ventana4, palabra, palabra_a_idx_4, idx_a_palabra_4, cantidad=3
        )

print("Resultados con ventana= 5")
for palabra in palabras_para_probar:
        mostrarPalabrasSimilares(
            W_ventana5, palabra, palabra_a_idx_5, idx_a_palabra_5, cantidad=3
        ) 
'''


r"""def contarFrecuencias(tokens):

    frecuencias_aboslutas = {}

    for token in tokens:
        if token not in frecuencias_aboslutas:
            frecuencias_aboslutas[token] = 1
        else:
            frecuencias_aboslutas[token] = frecuencias_aboslutas[token] + 1

    total = len(tokens)

    frecuencias_relativas = {}

    for palabra in frecuencias_aboslutas:
        frecuencias_relativas[palabra] = frecuencias_aboslutas[palabra] / total

    return frecuencias_relativas, frecuencias_aboslutas

def topNpalabras(frecuencias_relativas, n):
    palabras_ordenadas = sorted(frecuencias_relativas.items(), key=lambda elemento: elemento[1], reverse=True)
    return palabras_ordenadas[:n]

def graficarHistogramaPalabrasFrecuentas(palabras_ordenadas):
    palabras = [palabra[0] for palabra in palabras_ordenadas]
    frecuencias = [palabra[1] for palabra in palabras_ordenadas]

    plt.figure(figsize=(10, 6))
    plt.bar(palabras, frecuencias)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frecuencia relativa")
    plt.title(f"Top {len(palabras)} palabras más frecuentes")
    plt.tight_layout()
    plt.show()

tokens = cargaProcesarTexto(r"C:\Users\Ale Crespo\Desktop\aprendizaje-automatico-avanzado\tp1\tp1-prueba-aav.txt")
frecuencias_relativas, frecuencias_aboslutas = contarFrecuencias(tokens)
top_20_palabras = topNpalabras(frecuencias_relativas, n=20)
top_20_palabras_abs = topNpalabras(frecuencias_aboslutas, n=20)
graficarHistogramaPalabrasFrecuentas(top_20_palabras)
graficarHistogramaPalabrasFrecuentas(top_20_palabras_abs)
"""
tokens = cargaProcesarTexto(r"C:\Users\Ale Crespo\Desktop\aprendizaje-automatico-avanzado\tp1\tp1-aav.txt")

import pickle
import os

W_100, palabra_a_indice_100, indice_a_palabra_100 = entrenar(tokens, ventana=3, epocas=100, N=50, eta=0.05)

carpeta_script = os.path.dirname(os.path.abspath(__file__))

ruta_pesos = os.path.join(carpeta_script, "pesos_cbow_100epocas.npz")
np.savez(ruta_pesos, W=W_100)
print(f"Guardado en: {ruta_pesos}")

ruta_vocab = os.path.join(carpeta_script, "vocabulario_100epocas.pkl")
with open(ruta_vocab, "wb") as f:
    pickle.dump({"palabra_a_indice": palabra_a_indice_100, "indice_a_palabra": indice_a_palabra_100}, f)
print(f"Guardado en: {ruta_vocab}")

mostrarPalabrasSimilaresCoseno(W_100, "tiempo", palabra_a_indice_100, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_100, "tiempo", palabra_a_indice_100, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_100, "hombre", palabra_a_indice_100, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_100, "hombre", palabra_a_indice_100, cantidad=20)


"""W_15_copia, palabra_a_indice_15_copia, indice_a_palabra_15_copia = entrenar(tokens, ventana=3, epocas=15, N=50, eta=0.05)

carpeta_script = os.path.dirname(os.path.abspath(__file__))

ruta_pesos = os.path.join(carpeta_script, "pesos_cbow_15epocas.npz")
np.savez(ruta_pesos, W=W_15_copia)
print(f"Guardado en: {ruta_pesos}")

ruta_vocab = os.path.join(carpeta_script, "vocabulario_15epocas.pkl")
with open(ruta_vocab, "wb") as f:
    pickle.dump({"palabra_a_indice": palabra_a_indice_15_copia, "indice_a_palabra": indice_a_palabra_15_copia}, f)
print(f"Guardado en: {ruta_vocab}")

mostrarPalabrasSimilaresCoseno(W_15_copia, "tiempo", palabra_a_indice_15_copia, cantidad=10)
mostrarPalabrasSimilaresProdVectorial(W_15_copia, "tiempo", palabra_a_indice_15_copia, cantidad=10)

mostrarPalabrasSimilaresCoseno(W_15_copia, "hombre", palabra_a_indice_15_copia, cantidad=10)
mostrarPalabrasSimilaresProdVectorial(W_15_copia, "hombre", palabra_a_indice_15_copia, cantidad=10)"""

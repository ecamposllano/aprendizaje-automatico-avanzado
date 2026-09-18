import cupy as cp
import cupyx
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import time

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
    V = len(palabras_unicas)     # cantidad total de palbaras unicas

    palabra_a_indice = {}
    indice_a_palabra = {}

    numero = 0
    for palabra in palabras_unicas:
        palabra_a_indice[palabra] = numero
        indice_a_palabra[numero] = palabra
        numero = numero + 1

    return palabra_a_indice, indice_a_palabra, V


def generaContextos(tokens, palabra_a_indice, ventana=4):

    ejemplos = []   # Donde se almacenaran todos los pares generados de contextos e indices objetivos
    total_tokens = len(tokens)   # Cant total de tokens que tiene el corpus

    # Recorremos asegurando que haya suficiente margen a izquierda y derecha
    for i in range(ventana, total_tokens - ventana):
        # La palabra del medio es el objetivo
        palabra_objetivo = tokens[i]                             # la palabra en la posición i del texto
        indice_objetivo = palabra_a_indice[palabra_objetivo]     # su índice fijo en el vocabulario

        # Las palabras de alrededor son el contexto
        palabras_izq = tokens[i - ventana : i]
        palabras_der = tokens[i + 1 : i + ventana + 1]
        contexto = palabras_izq + palabras_der

        # Convertimos las palabras del contexto a sus índices
        c_indices = []
        for palabra in contexto:
            c_indices.append(palabra_a_indice[palabra])     # cada palabra del contexto -> su índice

        ejemplos.append((c_indices, indice_objetivo))   # (indices de las palabras del contexto, inidice de la palabra objetivo)
    return ejemplos     # lista de tuplas (c_indices, indice_objetivo), una por cada palabra central posible


def inicializarPesos(V, N, seed=42):
    cp.random.seed(seed)
    W = 0.1 * cp.random.randn(V, N)
    W_prima = 0.1 * cp.random.randn(N, V)

    return W, W_prima


def propagacion_batch(W, W_prima, batch_c_indices):
    #  B = tamaño del loye, C = cantidad de palabras de contexto
    B, C = batch_c_indices.shape
    N = W.shape[1]    # Cant de unidades en la capa oculat

    # Se extraen de W los vectores de cada palabra de contexto del lote, s suma esos
    # C vectores por cada ejemplo del batch y se promedia
    h = cp.sum(W[batch_c_indices], axis=1) / C

    # Producto matricial (B, N) @ (N, V) = (B, V)
    u = h @ W_prima

    # Softmax
    exp_u = cp.exp(u - cp.max(u, axis=1, keepdims=True))
    y = exp_u / cp.sum(exp_u, axis=1, keepdims=True)

    return h, y, u


def retropropagacion_batch(W, W_prima, batch_c_indices, batch_objetivos, h, y, eta):
    #  B = tamaño del loye, C = cantidad de palabras de contexto
    B, C = batch_c_indices.shape
    V = W_prima.shape[1]

    # Error de salida: e = y - t
    # Una fila para cada ejemplodel batch, para cada uno se resta 1 solo a la posición de su propia palabra objetivo
    e = y.copy()
    e[cp.arange(B), batch_objetivos] -= 1.0

    # Gradiente de W': dW' = h^T @ e
    dW_prima = h.T @ e / B

    # Error propagado hacia la capa oculta: EH = e @ W'^T
    EH = e @ W_prima.T

    # Actualizamos W'
    W_prima -= eta * dW_prima

    # Actualización de W
    for i in range(C):
        indices_columna = batch_c_indices[:, i]
        # cupy.scatter_add actualiza las filas repetidas en el batch
        cupyx.scatter_add(W, indices_columna, -(eta / C) * EH)
    return W, W_prima


def calcularPerdida_batch(y, batch_objetivos):
    B = y.shape[0]
    # Extraemos la probabilidad asignada a la palabra objetivo correcta para cada ejemplo del batch
    prob_correctas = y[cp.arange(B), batch_objetivos]
    # Evitamos log(0) sumando una constante muy pequeña epsilon
    perdida = -cp.log(prob_correctas + 1e-15)
    return cp.sum(perdida)


def similitudCoseno(v_A, v_B):

    norma_A = cp.linalg.norm(v_A)
    norma_B = cp.linalg.norm(v_B)

    if norma_A == 0 or norma_B == 0:
        return 0.0

    return cp.dot(v_A, v_B) / (norma_A * norma_B)


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
            # Guardamos primero la similitud
            puntajes.append((similitud, otra_palabra))

    # 3. Ordenamos de mayor a menor y cortamos las primeras
    puntajes.sort(reverse=True)
    mejores = puntajes[:cantidad]

    # 4. Mostramos los resultados
    print(f"\nPalabras más parecidas a '{palabra_buscada}':")
    for similitud, palabra in mejores:
        print(f"  - {palabra}: {similitud:.4f}")




def entrenar_por_batches(tokens, ventana, epocas, N, eta, batch_tamaño):
    palabra_a_indice, indice_a_palabra, V = armarDiccionario(tokens)
    ejemplos = generaContextos(tokens, palabra_a_indice, ventana=ventana)

    print(f"ENTRENANDO POR LOTES (Tamaño del lote = {batch_tamaño}) | Ventana = {ventana}")
    print(f"Tokens: {len(tokens)} | Vocabulario |V|: {V} | Ejemplos totales: {len(ejemplos)}")

    W, W_prima = inicializarPesos(V, N)

    # Convertimos los ejemplos a arreglos de CuPy
    c_indices_todos = cp.array([item[0] for item in ejemplos], dtype=cp.int32)
    objetivos_todos = cp.array([item[1] for item in ejemplos], dtype=cp.int32)

    num_ejemplos = len(ejemplos)

    for epoca in range(1, epocas + 1):
        inicio_epoca = time.time()
        perdida_total = 0.0

        # Mezclamos los índices en cada época
        indices_mezclados = cp.random.permutation(num_ejemplos)

        for i in range(0, num_ejemplos, batch_tamaño):
            batch_idxs = indices_mezclados[i:i + batch_tamaño]

            batch_c = c_indices_todos[batch_idxs]
            batch_obj = objetivos_todos[batch_idxs]

            h, y, u = propagacion_batch(W, W_prima, batch_c)
            perdida = calcularPerdida_batch(y, batch_obj)
            perdida_total += perdida

            W, W_prima = retropropagacion_batch(W, W_prima, batch_c, batch_obj, h, y, eta)

        perdida_promedio = float(perdida_total / num_ejemplos)
        duracion_epoca = time.time() - inicio_epoca
        print(f"Época {epoca:02d}/{epocas} - Pérdida: {perdida_promedio:.4f} - Duración: {duracion_epoca:.1f} seg")

    return cp.asnumpy(W), palabra_a_indice, indice_a_palabra



carpeta_script = os.path.dirname(os.path.abspath(__file__))

ruta_texto = os.path.join(carpeta_script, "tp1-aav.txt")
tokens = cargaProcesarTexto(ruta_texto)


"""def contarFrecuencias(tokens):

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

tokens = cargaProcesarTexto(ruta_texto)
frecuencias_relativas, frecuencias_aboslutas = contarFrecuencias(tokens)
top_20_palabras = topNpalabras(frecuencias_relativas, n=20)
top_20_palabras_abs = topNpalabras(frecuencias_aboslutas, n=20)
graficarHistogramaPalabrasFrecuentas(top_20_palabras)
graficarHistogramaPalabrasFrecuentas(top_20_palabras_abs)"""

"""W_final, palabra_a_indice, indice_a_palabra = entrenar_por_batches(tokens=tokens, ventana=4, epocas=100, N=50, eta=0.05, batch_size=128)


ruta_pesos = os.path.join(carpeta_script, "pesos_cbow.npz")
np.savez(ruta_pesos, W=W_final)
print(f"Guardado en: {ruta_pesos}")

ruta_vocab = os.path.join(carpeta_script, "vocabulario.pkl")

with open(ruta_vocab, "wb") as f:
    pickle.dump({"palabra_a_indice": palabra_a_indice, "indice_a_palabra": indice_a_palabra}, f)
print(f"Guardado en: {ruta_vocab}")


mostrarPalabrasSimilaresCoseno(W_final, "tiempo", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "tiempo", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "hombre", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "hombre", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "calle", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "calle", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "diez", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "diez", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "azul", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "azul", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "doctor", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "doctor", palabra_a_indice, cantidad=20) """


W_final, palabra_a_indice, indice_a_palabra = entrenar_por_batches(tokens=tokens, ventana=5, epocas=100, N=50, eta=0.05, batch_size=128)


ruta_pesos = os.path.join(carpeta_script, "pesos_cbow_5.npz")
np.savez(ruta_pesos, W=W_final)
print(f"Guardado en: {ruta_pesos}")

ruta_vocab = os.path.join(carpeta_script, "vocabulario_5.pkl")

with open(ruta_vocab, "wb") as f:
    pickle.dump({"palabra_a_indice": palabra_a_indice, "indice_a_palabra": indice_a_palabra}, f)
print(f"Guardado en: {ruta_vocab}")


mostrarPalabrasSimilaresCoseno(W_final, "tiempo", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "tiempo", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "hombre", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "hombre", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "calle", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "calle", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "diez", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "diez", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "azul", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "azul", palabra_a_indice, cantidad=20)

mostrarPalabrasSimilaresCoseno(W_final, "doctor", palabra_a_indice, cantidad=20)
mostrarPalabrasSimilaresProdVectorial(W_final, "doctor", palabra_a_indice, cantidad=20)


"""#Para entrenar con BPE, el corpus completo, vocabulario de 5000

from bpe import cargarTexto, entrenarBPE, fragmentarTexto, guardarBPE, cargarBPE

ruta_texto = os.path.join(carpeta_script, "tp1-aav.txt")
ruta_bpe   = os.path.join(carpeta_script, "bpe_merges.pkl")

# Si ya existe el archivo guardado, lo cargamos; si no, lo entrenamos una sola vez
if os.path.exists(ruta_bpe):
    merges, tokens = cargarBPE(ruta_bpe)
    print("BPE cargado exitosamente desde el archivo .pkl")
else:
    print("No hay BPE guardado: entrenando (solo esta vez)...")
    texto = cargarTexto(ruta_texto)
    merges, _ = entrenarBPE(texto, tamano_vocabulario=5000)
    tokens = fragmentarTexto(texto, merges)
    guardarBPE(ruta_bpe, merges, tokens)"""
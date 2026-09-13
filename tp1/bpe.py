
import collections
import pickle
import os



FIN_PALABRA = "</w>"   # marca el final de cada palabra


def cargarTexto(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        texto = archivo.read()
    return texto.lower()


def armarVocabPalabras(texto):
    vocab_palabras = {}
    for palabra in texto.split():
        palabra_separada = " ".join(list(palabra)) + " " + FIN_PALABRA
        if palabra_separada in vocab_palabras:
            vocab_palabras[palabra_separada] = vocab_palabras[palabra_separada] + 1
        else:
            vocab_palabras[palabra_separada] = 1
    return vocab_palabras




def contarPares(vocab_palabras):
    pares = collections.defaultdict(int)
    for palabra, frecuencia in vocab_palabras.items():
        simbolos = palabra.split()
        for i in range(len(simbolos) - 1):
            par = (simbolos[i], simbolos[i + 1])
            pares[par] = pares[par] + frecuencia
    return pares


def unirPar(par, vocab_palabras):
    a, b = par
    viejo = a + " " + b
    nuevo = a + b

    vocab_nuevo = {}
    for palabra, frecuencia in vocab_palabras.items():
        simbolos = palabra.split()
        resultado = []
        i = 0
        while i < len(simbolos):
            if i < len(simbolos) - 1 and simbolos[i] == a and simbolos[i + 1] == b:
                resultado.append(nuevo)
                i = i + 2
            else:
                resultado.append(simbolos[i])
                i = i + 1
        palabra_nueva = " ".join(resultado)
        if palabra_nueva in vocab_nuevo:
            vocab_nuevo[palabra_nueva] = vocab_nuevo[palabra_nueva] + frecuencia
        else:
            vocab_nuevo[palabra_nueva] = frecuencia
    return vocab_nuevo



def entrenarBPE(texto, tamano_vocabulario=500, min_frecuencia=2, verbose=True):   
    vocab_palabras = armarVocabPalabras(texto)

    # vocabulario inicial: todos los signos elementales
    tokens_vocabulario = set()
    for palabra in vocab_palabras:
        for simbolo in palabra.split():
            tokens_vocabulario.add(simbolo)

    if verbose:
        print(f"Signos iniciales: {len(tokens_vocabulario)}")

    merges = []
    while len(tokens_vocabulario) < tamano_vocabulario:
        pares = contarPares(vocab_palabras)
        if len(pares) == 0:
            break

        mejor_par = max(pares, key=pares.get)
        if pares[mejor_par] < min_frecuencia:
            break   # ya no hay pares suficientemente frecuentes

        vocab_palabras = unirPar(mejor_par, vocab_palabras)
        merges.append(mejor_par)
        tokens_vocabulario.add(mejor_par[0] + mejor_par[1])

        if verbose and len(merges) % 100 == 0:
            print(f"  merge {len(merges)}: {mejor_par} (freq {pares[mejor_par]}) "
                  f"| |V| = {len(tokens_vocabulario)}")

    if verbose:
        print(f"Merges realizados: {len(merges)} | Vocabulario final: {len(tokens_vocabulario)}")

    return merges, sorted(tokens_vocabulario)




def fragmentarPalabra(palabra, merges_ordenados):
    simbolos = list(palabra) + [FIN_PALABRA]

    for (a, b) in merges_ordenados:
        i = 0
        resultado = []
        while i < len(simbolos):
            if i < len(simbolos) - 1 and simbolos[i] == a and simbolos[i + 1] == b:
                resultado.append(a + b)
                i = i + 2
            else:
                resultado.append(simbolos[i])
                i = i + 1
        simbolos = resultado
    return simbolos


def fragmentarTexto(texto, merges, cache=None):
    if cache is None:
        cache = {}
    tokens = []
    for palabra in texto.lower().split():
        if palabra not in cache:
            cache[palabra] = fragmentarPalabra(palabra, merges)
        tokens.extend(cache[palabra])
    return tokens



def guardarBPE(ruta, merges, tokens):
    datos = {"merges": merges, "tokens": tokens}
    with open(ruta, "wb") as f:
        pickle.dump(datos, f)
    print(f"BPE guardado en: {ruta}")

def cargarBPE(ruta):
    with open(ruta, "rb") as f:
        datos = pickle.load(f)
    print(f"BPE cargado de: {ruta}")
    return datos["merges"], datos["tokens"]




if __name__ == "__main__":
    carpeta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_texto = os.path.join(carpeta_script, "tp1-aav.txt")
    ruta_bpe = os.path.join(carpeta_script, "bpe_merges.pkl")

    texto = cargarTexto(ruta_texto)

    V_palabras = len(set(texto.split()))
    print(f"Vocabulario en palabras: {V_palabras}")

    if os.path.exists(ruta_bpe):
        merges, tokens_bpe = cargarBPE(ruta_bpe)
        print("BPE ya existía, se cargó sin re-entrenar.")
    else:
        merges, _ = entrenarBPE(texto, tamano_vocabulario=5000, min_frecuencia=2)
        tokens_bpe = fragmentarTexto(texto, merges)
        guardarBPE(ruta_bpe, merges, tokens_bpe)

    print(f"Tokens BPE totales: {len(tokens_bpe)}")
    print(f"Vocabulario BPE real usado: {len(set(tokens_bpe))}")
    print("Primeros 40 tokens:", tokens_bpe[:40])
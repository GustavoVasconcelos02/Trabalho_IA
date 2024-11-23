from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
import nltk

# Baixar recursos do NLTK (apenas na primeira execução)
download('punkt')
download('stopwords')

# Limpa o texto removendo pontuações, stopwords e convertendo para minúsculas
def limpar_texto(texto):
    stop_words = set(stopwords.words('portuguese'))
    palavras = word_tokenize(texto.lower())
    palavras_limpa = [
        palavra for palavra in palavras
        if palavra.isalnum() and palavra not in stop_words
    ]
    return " ".join(palavras_limpa)

# Aplica a limpeza a uma lista de textos
def limpar_lista_textos(lista_textos):
    
    return [limpar_texto(texto) for texto in lista_textos]

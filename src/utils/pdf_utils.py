import os
from PyPDF2 import PdfReader

# Extrai o texto de um arquivo PDF
def pdf_para_txt(caminho_pdf):
    texto = ""
    try:
        with open(caminho_pdf, 'rb') as f:
            leitor = PdfReader(f)
            for pagina in leitor.pages:
                texto += pagina.extract_text() or ""
    except Exception as e:
        print(f"Erro ao processar o arquivo {caminho_pdf}: {e}")
    return texto

# Lê todos os PDFs nos diretórios fornecidos e retorna os textos e suas respectivas classes
def carregar_pdfs_de_diretorios(diretorios):
    textos = []
    classes = []
    for classe, caminho in diretorios.items():
        if not os.path.isdir(caminho):
            print(f"Diretório não encontrado: {caminho}")
            continue
        for arquivo in os.listdir(caminho):
            if arquivo.endswith('.pdf'):
                caminho_arquivo = os.path.join(caminho, arquivo)
                texto = pdf_para_txt(caminho_arquivo)
                if texto:  # Apenas adiciona PDFs com conteúdo
                    textos.append(texto)
                    classes.append(classe)
    return textos, classes

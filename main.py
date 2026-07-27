import os
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import pandas as pd

# --- Configuración ---
# Asegúrate de tener tu API Key en la variable de entorno OPENAI_API_KEY
# Ejemplo: export OPENAI_API_KEY="tu_api_key"

def cargar_documento(ruta):
    if ruta.endswith(".pdf"):
        reader = PdfReader(ruta)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()
        return texto
    elif ruta.endswith(".csv"):
        df = pd.read_csv(ruta)
        return df.to_string()
    else:
        print("Formato no soportado. Usa PDF o CSV.")
        sys.exit(1)

def crear_agente(texto):
    # Dividir texto en fragmentos
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(texto)

    # Crear embeddings y base vectorial
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Crear agente de preguntas y respuestas
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa

def main():
    ruta = "data/documento.pdf"  # Cambia a tu archivo CSV si lo prefieres
    texto = cargar_documento(ruta)
    agente = crear_agente(texto)

    print("🤖 Agente listo. Escribe tu pregunta (o 'salir' para terminar):")
    while True:
        pregunta = input(">> ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        respuesta = agente.run(pregunta)
        print(f"Respuesta: {respuesta}\n")

if __name__ == "__main__":
    main()

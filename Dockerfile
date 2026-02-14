# Usa uma imagem leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo da pasta atual para o container
COPY . .

# Expõe a porta que o Streamlit usa
EXPOSE 8501

# Comando para rodar o aplicativo quando o container iniciar
# O address 0.0.0.0 é obrigatório para funcionar via Docker
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
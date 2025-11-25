import os
from openai import OpenAI

# Configuração da API DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def configure_deepseek():
    """Configura o cliente OpenAI para usar a API DeepSeek"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    return client

def test_deepseek_connection():
    """Testa a conexão com a API DeepSeek"""
    try:
        client = configure_deepseek()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Olá, teste de conexão."}],
            max_tokens=10
        )
        return True, "Conexão com DeepSeek estabelecida com sucesso!"
    except Exception as e:
        return False, f"Erro na conexão com DeepSeek: {str(e)}"

# Testar a conexão ao importar
if __name__ == "__main__":
    success, message = test_deepseek_connection()
    print(message)

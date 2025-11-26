"""
Configuração centralizada para provedores de LLM.
As chaves de API são carregadas de variáveis de ambiente ou de um arquivo .env local.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI, BadRequestError
import anthropic


def _load_local_env():
    """Carrega variáveis de um .env local (gitignored) se existir, sem sobrescrever variáveis já definidas."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    
    try:
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"⚠️  Não foi possível carregar {env_path}: {exc}")


# Carregar .env antes de ler as variáveis
_load_local_env()


# Configurações dos provedores
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# URLs base
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class AbstractChatModel:
    """Interface base para modelos de chat."""
    
    def chat_completion(self, messages, temperature: float, max_tokens: int) -> str:
        """Envia uma query para o modelo de chat.

        :param messages: As mensagens para enviar ao modelo (formato OpenAI).
        :param temperature: A temperatura de amostragem.
        :param max_tokens: O número máximo de tokens a gerar.

        Returns:
            str: A resposta do modelo.
        """
        raise NotImplementedError


class OpenAIChatModel(AbstractChatModel):
    """Modelo de chat para OpenAI (GPT-4, GPT-3.5, etc.)"""
    
    def __init__(self, client: OpenAI, model: str):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature: float, max_tokens: int) -> str:
        def _send(temp_value, use_max_completion_tokens=True):
            """Helper para chamar a API com os parâmetros corretos."""
            kwargs = {
                "model": self.model,
                "messages": messages,
                "timeout": 120,
            }
            if temp_value is not None:
                kwargs["temperature"] = temp_value
            if use_max_completion_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            return self.client.chat.completions.create(**kwargs)

        try:
            response = _send(temperature, use_max_completion_tokens=True)
        except BadRequestError as exc:
            exc_str = str(exc)
            if "max_tokens" in exc_str:
                response = _send(temperature, use_max_completion_tokens=False)
            elif "temperature" in exc_str:
                response = _send(1, use_max_completion_tokens=True)
            else:
                raise
        
        try:
            response_content = response.choices[0].message.content
        except:
            print(f"Resposta inválida: {response}")
            response_content = ""
        
        if response_content is None:
            response_content = ""
        
        return response_content

    def __repr__(self) -> str:
        return f"OpenAI({self.model})"


class AnthropicChatModel(AbstractChatModel):
    """Modelo de chat para Anthropic Claude."""
    
    def __init__(self, client: anthropic.Anthropic, model: str):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature: float, max_tokens: int) -> str:
        # Converter formato OpenAI para formato Anthropic
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": anthropic_messages,
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            if temperature is not None:
                kwargs["temperature"] = temperature
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        
        except Exception as e:
            print(f"Erro na API Anthropic: {e}")
            return ""

    def __repr__(self) -> str:
        return f"Anthropic({self.model})"


class DeepSeekChatModel(AbstractChatModel):
    """Modelo de chat para DeepSeek."""
    
    def __init__(self, client: OpenAI, model: str):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature: float, max_tokens: int) -> str:
        def _send(temp_value, use_max_completion_tokens=True):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "timeout": 90,
            }
            if temp_value is not None:
                kwargs["temperature"] = temp_value
            if use_max_completion_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            return self.client.chat.completions.create(**kwargs)

        try:
            response = _send(temperature, use_max_completion_tokens=True)
        except BadRequestError as exc:
            exc_str = str(exc)
            if "max_tokens" in exc_str:
                response = _send(temperature, use_max_completion_tokens=False)
            elif "temperature" in exc_str:
                response = _send(1, use_max_completion_tokens=True)
            else:
                raise
        
        try:
            response_content = response.choices[0].message.content
        except:
            print(f"Resposta inválida: {response}")
            response_content = ""
        
        if response_content is None:
            response_content = ""
        
        return response_content

    def __repr__(self) -> str:
        return f"DeepSeek({self.model})"


# Mapeamento de modelos disponíveis (somente DeepSeek, GPT-5.1 e Claude Opus 4.1)
AVAILABLE_MODELS = {
    # OpenAI
    "gpt-5.1": {"provider": "openai", "display_name": "GPT-5.1", "speed": "fast"},

    # Anthropic Claude
    "claude-opus-4-20250514": {"provider": "anthropic", "display_name": "Claude Opus 4 (2025-05-14)", "speed": "slow"},

    # DeepSeek
    "deepseek-chat": {"provider": "deepseek", "display_name": "DeepSeek Chat (Econômico)", "speed": "slow"},
    "deepseek-coder": {"provider": "deepseek", "display_name": "DeepSeek Coder", "speed": "slow"},
}


def setup_openai(model: str) -> OpenAIChatModel:
    """Configura um modelo OpenAI."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY não configurada. Execute: export OPENAI_API_KEY='sua-chave'")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    return OpenAIChatModel(client, model)


def setup_anthropic(model: str) -> AnthropicChatModel:
    """Configura um modelo Anthropic Claude."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY/CLAUDE_API_KEY não configurada. Exporte ANTHROPIC_API_KEY='sua-chave'")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return AnthropicChatModel(client, model)


def setup_deepseek(model: str) -> DeepSeekChatModel:
    """Configura um modelo DeepSeek."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY não configurada. Execute: export DEEPSEEK_API_KEY='sua-chave'")
    
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return DeepSeekChatModel(client, model)


def setup(model: str) -> AbstractChatModel:
    """Configura um modelo de chat baseado no nome, limitado aos provedores suportados.
    
    Args:
        model: Nome do modelo (ex: 'gpt-5.1', 'claude-opus-4-20250514', 'deepseek-chat')
    
    Returns:
        AbstractChatModel: Instância do modelo configurado
    """
    if isinstance(model, AbstractChatModel):
        return model

    if model not in AVAILABLE_MODELS:
        supported = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Modelo '{model}' não suportado. Modelos permitidos: {supported}")

    provider = AVAILABLE_MODELS[model]["provider"]
    if provider == "openai":
        return setup_openai(model)
    if provider == "anthropic":
        return setup_anthropic(model)
    if provider == "deepseek":
        return setup_deepseek(model)

    raise ValueError(f"Provedor '{provider}' não suportado para o modelo {model}")


def get_available_models() -> dict:
    """Retorna lista de modelos disponíveis com status de configuração."""
    models = {}
    
    for model_id, info in AVAILABLE_MODELS.items():
        provider = info["provider"]
        available = False
        
        if provider == "openai" and OPENAI_API_KEY:
            available = True
        elif provider == "anthropic" and ANTHROPIC_API_KEY:
            available = True
        elif provider == "deepseek" and DEEPSEEK_API_KEY:
            available = True
        
        models[model_id] = {
            **info,
            "available": available
        }
    
    return models


def check_api_keys() -> dict:
    """Verifica quais chaves de API estão configuradas."""
    return {
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY),
    }


# Função de compatibilidade com o código existente
import copy

# Importar cache de LLM
try:
    from t2ebm.cache import get_llm_cache
    LLM_CACHE_AVAILABLE = True
except ImportError:
    LLM_CACHE_AVAILABLE = False
    print("Warning: LLM cache not available")


def chat_completion(llm, messages, use_cache: bool = True):
    """Executa uma sequência de mensagens com um AbstractChatModel.
    
    Compatível com a interface do t2ebm.llm.chat_completion.
    
    Args:
        llm: O modelo LLM ou string com nome do modelo
        messages: Lista de mensagens no formato OpenAI
        use_cache: Se True, usa cache para respostas LLM (padrão: True)
    """
    llm_instance = setup(llm)
    model_name = str(llm_instance) if hasattr(llm_instance, '__str__') else str(llm)
    messages = copy.deepcopy(messages)
    
    # Obter cache se disponível
    cache = get_llm_cache() if LLM_CACHE_AVAILABLE and use_cache else None
    
    for msg_idx in range(len(messages)):
        if messages[msg_idx]["role"] == "assistant":
            if "content" not in messages[msg_idx]:
                # Preparar contexto para cache
                context_messages = messages[:msg_idx]
                cache_kwargs = {
                    "temperature": messages[msg_idx].get("temperature", 0.7),
                    "max_tokens": messages[msg_idx].get("max_tokens", 1000),
                }
                
                # Tentar obter do cache
                cached_response = None
                if cache:
                    cached_response = cache.get_cached_response(
                        model_name, 
                        context_messages,
                        **cache_kwargs
                    )
                
                if cached_response:
                    messages[msg_idx]["content"] = cached_response
                else:
                    # Fazer chamada ao LLM
                    response = llm_instance.chat_completion(
                        context_messages,
                        temperature=cache_kwargs["temperature"],
                        max_tokens=cache_kwargs["max_tokens"],
                    )
                    messages[msg_idx]["content"] = response
                    
                    # Salvar no cache
                    if cache and response:
                        cache.set_cached_response(
                            model_name,
                            context_messages,
                            response,
                            **cache_kwargs
                        )
            
            # Remover chaves extras
            keys = list(messages[msg_idx].keys())
            for k in keys:
                if k not in ["role", "content"]:
                    messages[msg_idx].pop(k)
    
    return messages


def test_connection(model: str = "deepseek-chat") -> bool:
    """Testa a conexão com um modelo."""
    try:
        llm = setup(model)
        response = llm.chat_completion(
            messages=[{"role": "user", "content": "Teste de conexão. Responda apenas 'OK'."}],
            temperature=0.1,
            max_tokens=10
        )
        print(f"✅ Conexão com {model} estabelecida: {response[:50]}")
        return True
    except Exception as e:
        print(f"❌ Erro ao conectar com {model}: {e}")
        return False


if __name__ == "__main__":
    print("=== Status das Chaves de API ===")
    keys = check_api_keys()
    for provider, configured in keys.items():
        status = "✅ Configurada" if configured else "❌ Não configurada"
        print(f"{provider}: {status}")
    
    print("\n=== Modelos Disponíveis ===")
    models = get_available_models()
    for model_id, info in models.items():
        status = "✅" if info["available"] else "❌"
        print(f"{status} {info['display_name']} ({model_id})")

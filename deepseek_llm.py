"""
Módulo personalizado para integração com DeepSeek API
Compatível com a interface do TalkToEBM
"""

from dataclasses import dataclass
from openai import OpenAI, BadRequestError
import copy
import os
from typing import Union

# Configuração da API DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

@dataclass
class AbstractChatModel:
    def chat_completion(self, messages, temperature: float, max_tokens: int):
        """Send a query to a chat model.

        :param messages: The messages to send to the model. We use the OpenAI format.
        :param temperature: The sampling temperature.
        :param max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The model response.
        """
        raise NotImplementedError


class DeepSeekChatModel(AbstractChatModel):
    client: OpenAI = None
    model: str = None

    def __init__(self, client, model):
        super().__init__()
        self.client = client
        self.model = model

    def chat_completion(self, messages, temperature, max_tokens):
        def _send(temp_value, use_max_completion_tokens=True):
            """Helper to call the API with the right param names."""
            kwargs = {
                "model": self.model,
                "messages": messages,
                "timeout": 90,
            }
            # Some models only accept the default temperature. Leave it out when None.
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
            # Retry with legacy max_tokens if the model does not support max_completion_tokens.
            if exc.code == "unsupported_parameter" and "max_tokens" in exc_str:
                response = _send(temperature, use_max_completion_tokens=False)
            # Some lightweight models do not support non-default temperature values.
            elif exc.code == "unsupported_value" and "temperature" in exc_str:
                response = _send(1, use_max_completion_tokens=True)
            else:
                raise
        # we return the completion string or "" if there is an invalid response/query
        try:
            response_content = response.choices[0].message.content
        except:
            print(f"Invalid response {response}")
            response_content = ""
        if response_content is None:
            print(f"Invalid response {response}")
            response_content = ""
        return response_content

    def __repr__(self) -> str:
        return f"{self.model}"


def deepseek_setup(model: str = "deepseek-chat"):
    """Setup a DeepSeek language model.

    :param model: The name of the model (e.g. "deepseek-chat").

    Returns:
        DeepSeekChatModel: A DeepSeek LLM interface
    """
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )

    # the llm
    return DeepSeekChatModel(client, model)


def setup(model: Union[AbstractChatModel, str]):
    """Setup a chat model. If the input is a string, we assume that it is the name of an DeepSeek model."""
    if isinstance(model, str):
        if model.startswith("deepseek"):
            model = deepseek_setup(model)
        else:
            # Fallback to original OpenAI setup
            from t2ebm.llm import openai_setup
            model = openai_setup(model)
    return model


# Also patch the setup function in t2ebm.llm to use our version
import t2ebm.llm
t2ebm.llm.setup = setup


def chat_completion(llm: Union[str, AbstractChatModel], messages):
    """Execute a sequence of user and assistant messages with an AbstractChatModel.

    Sends multiple individual messages to the AbstractChatModel.
    """
    llm = setup(llm)
    # we sequentially execute all assistant messages that do not have a content.
    messages = copy.deepcopy(messages)  # do not alter the input
    for msg_idx in range(len(messages)):
        if messages[msg_idx]["role"] == "assistant":
            if not "content" in messages[msg_idx]:
                # send message
                messages[msg_idx]["content"] = llm.chat_completion(
                    messages[:msg_idx],
                    temperature=messages[msg_idx]["temperature"],
                    max_tokens=messages[msg_idx]["max_tokens"],
                )
            # remove all keys except "role" and "content"
            keys = list(messages[msg_idx].keys())
            for k in keys:
                if not k in ["role", "content"]:
                    messages[msg_idx].pop(k)
    return messages


# Test function
def test_deepseek():
    """Test DeepSeek connection"""
    try:
        llm = deepseek_setup("deepseek-chat")
        response = llm.chat_completion(
            messages=[{"role": "user", "content": "Olá, teste de conexão com DeepSeek."}],
            temperature=0.7,
            max_tokens=50
        )
        print("✅ Conexão com DeepSeek estabelecida com sucesso!")
        print(f"Resposta: {response}")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão com DeepSeek: {e}")
        return False


if __name__ == "__main__":
    test_deepseek()

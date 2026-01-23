from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

ClientType = BaseChatModel
ValidationSetType = list[dict]
MetadataType = dict[str, Any]
ScoreType = float
PromptContentType = str

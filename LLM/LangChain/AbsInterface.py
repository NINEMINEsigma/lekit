from abc                                    import *
from typing                                 import *
from langchain_community.chat_models        import *
from langchain_core.language_models.base    import *
from langchain_core.tools                   import tool as function_call_tool

from lekit.File.Core                        import tool_file

# Abstract part Definition

class abs_llm_callable(Callable[[str], BaseMessage], ABC):
    @abstractmethod
    def __call__(self, message:str) -> BaseMessage:
        return None

class abs_llm_core(abs_llm_callable, ABC):
    @abstractmethod
    def save_hestroy(self, file:Union[tool_file,str]):
        pass
    @abstractmethod
    def load_hestory(self, file:Union[tool_file,str]):
        pass
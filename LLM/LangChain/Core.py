from langchain_community.chat_models    import *
from typing                             import *
from lekit.Str.Core                     import UnWrapper
from lekit.File.Core                    import tool_file
from lekit.Lang.CppLike                 import *

from langchain_core.language_models.base    import *
from langchain_core.prompts                 import ChatPromptTemplate

MessageObject = LanguageModelInput
MessageType = Literal[
    "human", 
    "user",
    "ai",
    "assistant",
    "system", 
    "function",
    "tool"
    ]

def do_make_content(**kwargs) -> MessageObject:
    result:Dict[str,str]={}
    for key in kwargs:
        result[key] = str(kwargs[key])
    return result
def make_content(role:MessageType, content:str):
    return do_make_content(role=role, content=content)
def make_system_prompt(prompt:str):
    return make_content('system', prompt)
def make_human_prompt(message:str):
    return make_content('human', message)
def make_assistant_prompt(message:str):
    return make_content('assistant', message)

class abs_core(ABC):
    @abstractmethod
    def __call__(self, message:str) -> BaseMessage:
        return None

class light_llama_core:
    def __init__(self,
                 model:         Union[str, tool_file, ChatLlamaCpp],
                 init_message:  Union[MessageObject, List[MessageObject]]   = []
                 ):
        if isinstance(init_message, List) is False:
            if isinstance(init_message, str):
                init_message = make_system_prompt(init_message)
            else:
                init_message = [init_message]
        if isinstance(model, ChatLlamaCpp) is False:
            model = ChatLlamaCpp(model_path=UnWrapper(model))
        self.init_message_list      :List[MessageObject]    = init_message
        self.hestroy_message_list   :List[MessageObject]    = self.init_message_list
        self.model                  :ChatLlamaCpp           = model
        self.last_result            :BaseMessage            = None
        
    def __str__(self):
        return str(self.hestroy_message_list)
    def __call__(self, message:str):
        self.hestroy_message_list.append(make_human_prompt(message))
        result = self.model.invoke(self.hestroy_message_list)
        self.hestroy_message_list.append(make_assistant_prompt(result.content))
        self.last_result = result
        return result
    
    def clear_hestroy(self):
        self.hestroy_message_list = self.init_message_list
        return self
    def pop_hestroy(self):
        result = self.hestroy_message_list.pop()
        return result
    def get_hestroy(self):
        return self.hestroy_message_list
    def set_hestroy(self, message:List[MessageObject]):
        self.hestroy_message_list = message
    def get_last_message(self):
        return self.hestroy_message_list[-1]
    def append_message(self, message:MessageObject):
        self.hestroy_message_list.append(message)
        return self

    def save_hestroy(self, file:Union[tool_file,str]):
        if isinstance(file, tool_file) is False:
            file = tool_file(UnWrapper(file))
            file.open('wb')
        file.data = self.hestroy_message_list
        
class light_prompt:
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            (
                "human", 
                "{input}"
            ),
        ]
    )

    ...

    chain = prompt | llm
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    '''
    def __init__(self):
        self.prompt:List[BaseMessage] = []
        
    def from_single_chat(self, system_format_prompt:str, human_format_asker:str):
        self.prompt.append(ChatPromptTemplate.from_messages(make_list(
            make_system_prompt(system_format_prompt),
            make_human_prompt(human_format_asker)
        )))
        return self
    
    def __call__(self, system_format_prompt:str, human_format_asker:str):
        return self.from_single_chat(system_format_prompt,human_format_asker)
    
    def invork(self, )
    
    
if __name__ == '__main__':
    # 设置模型路径
    model_path=r'D:\LLM\MODELs\llama3-8B\Meta-Llama-3-8B-Instruct\Meta-Llama-3-8B-Instruct-Q4_0.gguf'
    
    llm = light_llama_core(
        model=model_path,
        init_message=[make_system_prompt("You are a helpful assistant that translates English to Chinese. Translate the user sentence.")]
    )
    result = llm(r"What is the time now.")
    print("result:")
    print(result.content)
    print("llm hestroy:")
    print(llm.hestroy_message_list)
    
    print()
    
    result = llm(r"I am human.")
    print("result:")
    print(result.content)
    print("llm hestroy:")
    print(llm.hestroy_message_list)
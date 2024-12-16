from langchain_community.chat_models    import *
from typing                             import *
from lekit.Str.Core                     import UnWrapper
from lekit.File.Core                    import tool_file
from lekit.Lang.CppLike                 import *

from langchain_core.language_models.base    import *
from langchain_core.prompts                 import ChatPromptTemplate
from langchain_core.messages.utils          import convert_to_messages

from lekit.LLM.LangChain.AbsInterface   import abs_llm_core, abs_llm_callable

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

class light_llama_core(abs_llm_core):
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
    @override
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
    def append_hestroy(self, message:MessageObject):
        self.hestroy_message_list.append(message)
        return self
    @override
    def save_hestroy(self, file:Union[tool_file,str]):
        if isinstance(file, tool_file) is False:
            file = tool_file(UnWrapper(file))
            file.open('wb')
        file.data = self.hestroy_message_list
    @override
    def load_hestory(self, file:Union[tool_file,str]):
        if isinstance(file, tool_file) is False:
            file = tool_file(UnWrapper(file))
        if file.exists() is False:
            raise FileExistsError(f"{file.get_path()} is not found")
        file.open('rb')
        file.refresh()
        self.hestroy_message_list = file.data
    
    def get_last_message(self):
        return self.hestroy_message_list[-1]
    def append_message(self, message:MessageObject):
        self.hestroy_message_list.append(message)
        return self

class light_llama_prompt_core(Callable[[dict], BaseMessage]):
    def __init__(self, core:light_llama_core, prompt:ChatPromptTemplate):
        self.core:          light_llama_core    = core
        self.prompt:        ChatPromptTemplate  = prompt
        self.last_result:   BaseMessage         = None
        
    @override
    def __call__(self, message_inserter:dict) -> BaseMessage:
        chain = self.prompt|self.core
        self.last_result = chain.invoke(message_inserter)
        return self.last_result.content
        
class light_llama_prompt(Callable[[str, str], object]):
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
        self.prompt:List[MessageLikeRepresentation] = []
        
    def from_single_chat(self, role:MessageType, message_format:str):
        self.prompt.append((role,message_format))
        return self
    
    def __call__(self, role:MessageType, message_format:str):
        return self.from_single_chat(role,message_format)
    
    def invoke(self, core:light_llama_core):
        return light_llama_prompt_core(
            core, 
            ChatPromptTemplate.from_messages(self.prompt)
            )
    
    
if __name__ == '__main__':
    # 设置模型路径
    model_path=r'D:\LLM\MODELs\llama3-8B\Meta-Llama-3-8B-Instruct\Meta-Llama-3-8B-Instruct-Q4_0.gguf'
    
    llm = light_llama_core(
        model=model_path,
        init_message=[make_system_prompt("You are a helpful assistant that translates English to Chinese. Translate the user sentence.")]
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

    chain = prompt | llm.model
    print()
    print(chain.invoke(
        {
            "input_language": "English",
            "output_language": "Chinese",
            "input": "I love programming.",
        }
    ).content)
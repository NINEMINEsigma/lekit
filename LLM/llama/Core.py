from typing import Dict, List, Union
from lekit.File.Core import tool_file
from llama_cpp import Llama

class light_llama(Llama):
    def __init__(
        self,
        model_path: Union[str, tool_file],
        # Extra Params
        *args,
        **kwargs,  # type: ignore
    ):
        super().__init__(model_path=str(model_path), *args, **kwargs)
        self.function_pool:Dict[function] = None
        
    def __call__(self, prompt:str, *args, **kwargs):
        return super().__call__(prompt, *args, **kwargs)["choices"]["text"]
    
    
    
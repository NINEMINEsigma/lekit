from typing             import *
from lekit.File.Core    import tool_file
from lekit.Str.Core     import UnWrapper as UnWrapper2Str

import os

try:
    import numpy as np
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    import pandas as pd
except ImportError:
    pass

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
except ImportError:
    pass

const_config_file = "config.json"

class GlobalConfig:
    def get_config_file(self):
        return self.data_dir|self.__const_config_file
    
    @property
    def config_file(self):
        return self.get_config_file()
    
    def __init__(self, data_dir:Optional[Union[tool_file, str]]=None, is_try_create_data_dir:bool=False):
        # build up data folder
        if data_dir is None:
            data_dir = tool_file(os.path.abspath('.'))
        self.data_dir:tool_file = data_dir if isinstance(data_dir, tool_file) else tool_file(UnWrapper2Str(data_dir))
        if self.data_dir.exists() is False:
            if is_try_create_data_dir:
                self.data_dir.try_create_parent_path()
            else:
                raise FileNotFoundError(f"Can't find data dir: {self.data_dir.get_dir()}")
        if self.data_dir.is_dir() is False:
            self.data_dir.back_to_parent_dir()
        # build up init data file
        self.__data_pair:Dict[str, Any] = {}
        global const_config_file
        self.__const_config_file = const_config_file
        config_file = self.config_file
        if config_file.exists() is False:
            config_file.create()
        else:
            self.load_properties()
    def __del__(self):
        #self.save_properties()
        pass

    def save_temp_data(self, any_data):
        self.data_dir.data = any_data
        self.data_dir.save()
    def load_temp_data(self):
        return self.data_dir.load()
    
    def get_file(self, file:str, is_must:bool=False):
        result = self.data_dir|file
        if is_must and result.exists() is False:
            result.must_exists_path()
        return result
    def erase_file(self, file:str):
        result = self.data_dir|file
        if result.exists():
            try:
                with open(result.get_path(), "wb") as _:
                    return True
            except:
                pass
        return False
    def remove_file(self, file:str):
        result = self.data_dir|file
        if result.exists():
            try:
                result.remove()
                return True
            except:
                pass
        return False
    def create_file(self, file:str):
        result = self.data_dir|file
        if result.exists():
            return False
        if result.back_to_parent_dir().exists() is False:
            return False
        result.create()
        return True
    
    def __setitem__(self, key:str, value:str):
        self.__data_pair[key] = value
    def __getitem__(self, key:str):
        return self.__data_pair[key]
    def __contains__(self, key:str):
        return key in self.__data_pair
    def __delitem__(self, key:str):
        del self.__data_pair[key]
    def __iter__(self):
        return iter(self.__data_pair)
    def __len__(self):
        return len(self.__data_pair)
    
    def save_properties(self):
        config = self.config_file
        config.open('w', encoding='utf-8')
        config.data = {
            "properties": self.__data_pair
        }
        config.save()
        return self
    def load_properties(self):
        config = self.config_file
        if config.exists() is False:
            self.__data_pair = {}
        else:
            config.load_as_json()
            if "properties" in config.data:
                for property_name in config.data["properties"]:
                    self.__data_pair[property_name] = config.data["properties"][property_name]
            else:
                raise ValueError("Can't find properties in config file")
        return self    
                
    def print_source_pair(self):
        print(self.__data_pair)
                
    def Log(self, message_type:str, message):
        print(f"{message_type}: {UnWrapper2Str(message)}")
        return self
    def LogMessage(self, message:str):
        self.Log("Message", message)
        return self
    def LogWarning(self, message:str):
        self.Log("Warning", message)
        return self
    def LogError(self, message:str):
        self.Log("Error", message)
        return self
    def LogPropertyNotFound(self, message):
        self.Log("Property not found", message)
        return self
                
    def FindItem(self, key:str):
        if key in self.__data_pair:
            return self.__data_pair[key]
        else:
            self.LogPropertyNotFound(key)
            return None
                
                
                
                
                
                
                
                
                
                
                
                
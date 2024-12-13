import json
import csv
import xml.etree.ElementTree as ET
import os
import sys

from pydub import AudioSegment

from PIL import Image

from docx import Document
from docx.document import Document as DocumentObject

audio_file_type = ["mp3","ogg","wav"]
image_file_type = ['png', 'jpg', 'jpeg', 'bmp', 'svg', 'ico']

is_binary_file_functional_test_length:int = 1024

def is_binary_file(file_path: str) -> bool:
    try:
        global is_binary_file_functional_test_length
        with open(file_path, 'rb') as f:
            chunk = f.read(is_binary_file_functional_test_length)  # 读取文件的前缀字节
            return b'\x00' in chunk  # 如果包含NUL字符，则认为是二进制文件
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def get_extension_name(file:str):
        return os.path.splitext(file)[1][1:]
    
def is_image_file(file_path:str):
    return get_extension_name(file_path) in image_file_type

class tool_file:
    def __init__(self, file_path:str, file_mode:str=None):
        self.__file_path = file_path
        if file_mode is None:
            self.__file = None
        else:
            self.open(file_mode)
    def __del__(self):
        self.close()
    def __str__(self):
        return self.get_path()
        
    def create(self):
        if self.exists() == False:
            self.open('w')
            self.close()
    def exists(self):
        return os.path.exists(self.__file_path)
    def remove(self):
        self.close()
        if self.exists():
            os.remove(self.__file_path)

    def refresh(self):
        self.load()
        return self
    def open(self, mode='r', is_refresh=False, encoding:str='utf-8'):
        self.close()
        self.__file = open(self.__file_path, mode, encoding=encoding)
        if is_refresh:
            self.refresh()
        return self.__file
    def close(self):
        if self.__file:
            self.__file.close()
        return self.__file
    def is_open(self)->bool:
        return self.__file
        
    def load(self):
        suffix = self.get_extension()
        if suffix == 'json':
            self.load_as_json()
        elif suffix == 'csv':
            self.load_as_csv()
        elif suffix == 'xml':
            self.load_as_xml()
        elif suffix == 'xlsx' or suffix == 'xls':
            self.load_as_excel()
        elif suffix == 'txt':
            self.load_as_text()
        elif suffix == 'docx':
            self.load_as_docx()
        elif suffix in audio_file_type:
            self.load_as_audio()
        elif is_binary_file(self.__file_path):
            self.load_as_binary()
        elif is_image_file(self.__file_path):
            self.load_as_image()
        else:
            self.load_as_text()
        return self.data
    def load_as_json(self):
        self.open('r')
        self.data = json.load(self.__file)
        return self.data
    def load_as_csv(self):
        self.open('r')
        self.data = csv.reader(self.__file)
        return self.data
    def load_as_xml(self):
        self.open('r')
        self.data = ET.parse(self.__file)
        return self.data
    def load_as_dataframe(self):
        self.open('r')
        self.data = pd.read_csv(self.__file)
        return self.data
    def load_as_excel(self):
        self.open('r')
        self.data = pd.read_excel(self.__file)
        return self.data
    def load_as_binary(self):
        self.open('rb')
        self.data = self.__file.read()
        return self.data
    def load_as_text(self):
        self.open('r')
        self.data = self.__file.readlines()
        return self.data
    def load_as_wav(self):
        self.data = AudioSegment.from_wav(self.__file_path)
        return self.data
    def load_as_audio(self):
        self.data = AudioSegment.from_file(self.__file_path)
        return self.data
    def load_as_image(self):
        self.data = Image.open(self.__file_path)
        return self.data
    def load_as_docx(self):
        self.data = Document(self.__file_path)
        return self.data

    def save(self, path:str=None):
        suffix = self.get_extension(path)
        if suffix == 'json':
            self.save_as_json(path)
        elif suffix == 'csv':
            self.save_as_csv(path)
        elif suffix == 'xml':
            self.save_as_xml(path)
        elif suffix == 'xlsx' or suffix == 'xls':
            self.save_as_excel(path)
        elif suffix == 'txt':
            self.save_as_text(path)
        elif suffix == 'docx':
            self.save_as_docx(path)
        elif suffix in audio_file_type:
            self.save_as_audio(path, suffix)
        elif is_binary_file(self.__file_path):
            self.save_as_binary(path)
        elif is_image_file(self.__file_path):
            self.save_as_image(path)
        else:
            self.save_as_text(path)
        return self
    def save_as_json(self, path:str):
        if path is not None:
            with open(path, 'w') as f:
                json.dumps(self.data, f)
        else:
            json.dump(self.data, self.__file)
        return self
    def save_as_csv(self, path:str):
        if path is not None:
            with open(path, 'w') as f:
                csv.writer(f).writerows(self.data)
        else:
            csv.writer(self.__file).writerows(self.data)
        return self
    def save_as_xml(self, path:str):
        if path is not None:
            with open(path, 'w') as f:
                self.data.write(f)
        else:
            self.data.write(self.__file)
        return self
    def save_as_dataframe(self, path:str):
        if path is not None:
            self.data.to_csv(path)
        else:
            self.data.to_csv(self.__file)
        return self
    def save_as_excel(self, path:str):
        if path is not None:
            self.data.to_excel(path, index=False)
        else:
            self.data.to_excel(self.__file, index=False)
        return self
    def save_as_binary(self, path:str):
        if path is not None:
            with open(path, 'wb') as f:
                f.write(self.data)
        else:
            self.__file.write(self.data)
        return self
    def save_as_text(self, path:str):
        if path is not None:
            with open(path, 'w') as f:
                f.writelines(self.data)
        else:
            self.__file.writelines(self.data)
        return self
    def save_as_audio(self, path:str):
        self.data.export(path if path else self.__file_path, format=self.get_extension(path))
        return self
    def save_as_image(self, path:str):
        self.data.save(path if path else self.__file_path)
        return self
    def save_as_docx(self, path:str):
        if self.data is str:
            self.data = Document()
            table = self.data.add_table(rows=1, cols=1)
            table.cell(0, 0).text = self.data
        self.data.save(path if path else self.__file_path)
        return self
    
    def get_data_type(self):
        return type(self.data)
    def get_extension(self, path:str=None):
        path = path if path is not None else self.__file_path
        return get_extension_name(path)
    def get_path(self):
        return self.__file_path
    def get_filename(self):
        return os.path.basename(self.__file_path)
    
    def is_dir(self):
        return os.path.isdir(self.__file_path)
    def is_file(self):
        return os.path.isfile(self.__file_path)
    def is_binary_file(self):
        return is_binary_file(self.__file)
    def is_image(self):
         return is_image_file(self.__file_path)
    
    def try_create_parent_path(self):
        dir_path = os.path.dirname(self.__file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    def dir_iter(self):
        return os.listdir(self.__file_path)
    
    def append_text(self, line:str):
        if self.data is str:
            self.data = self.data + line
        elif self.data is DocumentObject:
            self.data.add_paragraph(line)
        else:
            raise TypeError(f"Unsupported data type for {sys._getframe().f_code.co_name}")
        return self
    
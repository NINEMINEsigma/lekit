from bs4 import BeautifulSoup as base

class light_bs(object):
    def __init__(self, html:str):
        self.__base = base(html, 'html.parser')
        
    def retarget(self,html:str):
        self.__base = base(html, 'html.parser')

    def first(self, tag, attrs=None, recursive=True, text=None, **kwargs):
        return self.__base.find(tag, attrs, recursive, text, **kwargs)
    
    def find(self, tag, attrs=None, recursive=True, text=None, **kwargs):
        return self.__base.find(tag, attrs, recursive, text, **kwargs)

    def find_all(self, tag, attrs=None, recursive=True, text=None, limit=None, **kwargs):
        return self.__base.find_all(tag, attrs, recursive, text, limit, **kwargs)

    def select(self, selector):
        return self.__base.select(selector)

    def text(self):
        return self.__base.text
    
if __name__ == "__main__":
    core = light_bs()

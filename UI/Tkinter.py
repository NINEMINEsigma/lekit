from typing import *
import tkinter as base

class light_tk_window:
    def __init__(self):
        self.window:    base.Tk = base.Tk()
        self.__title:   str     = "window"
        
    def configure(
        self,
        cnf:    Optional[Dict[str, Any]]    = None,
        **kwargs
        ) -> Optional[Dict[str, tuple[str, str, str, Any, Any]]]:
        return self.window.configure(cnf, **kwargs)
    def set_background_color(self, color):
        self.window.configure(bg=color)
    def set_foreground_color(self, color):
        self.window.configure(fg=color)

    @property
    def title(self):
        return self.__title
    @title.setter
    def title(self, title):
        self.__title = title
        self.window.title(title)
        return self.__title

    def add_label(self, text, position):
        label = base.Label(self.window, text=text)
        label.pack(position)
    def add_button(self, text, command, position):
        button = base.Button(self.window, text=text, command=command)
        button.pack(position)
    def add_input_field(self, position):
        entry = base.Entry(self.window)
        entry.pack(position)
        return entry

    def run(self):
        self.window.mainloop()

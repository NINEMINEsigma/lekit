from typing                                                 import *
from lekit.MathEx.Transform                                 import *
from lekit.File.Core                                        import tool_file, UnWrapper as Unwrapper2Str
from selenium                                               import webdriver
from selenium.webdriver.remote.webdriver                    import WebDriver
from selenium.webdriver.remote.webelement                   import WebElement
from selenium.webdriver.common.by                           import By, ByType
from selenium.webdriver.common.action_chains                import ActionChains as Mouse
from selenium.webdriver.common.keys import Keys             as     KeyBoard
from selenium.webdriver.support.ui import WebDriverWait     as     WaitTask
from selenium.webdriver.support import expected_conditions  as     EC
import                                                             time
import                                                             unittest

no_wait_enable_constexpr_value = 0
no_wait_enable_type = Literal[0]
if_wait_enable_constexpr_value = 1
if_wait_enable_type = Literal[1]
implicitly_wait_enable_constexpr_value = 2
implicitly_wait_enable_type = Literal[2]
wait_enable_type = Literal[
    no_wait_enable_type,
    if_wait_enable_type,
    implicitly_wait_enable_type
    ]

class selunit:
    """ instance of selenium functions """

    #web browser driver(default is MSedge)
    # "" __browser ""
    def __init__(
            self,
            browser:    WebDriver   = webdriver.Edge(),
            transform:  abs_box     = Rect(int(50), int(50), int(900), int(800)),
            delay = 3):
        if browser is None:
            raise ValueError("browser is none")
        self.__browser:                 WebDriver                           = browser
        self.__transform:               abs_box                             = transform
        self.delay:                     float                               = delay
        self.delay_notify:              float                               = True
        self.wait_enable:               wait_enable_type                    = no_wait_enable_type
        self.__if_wait_pred:            Callable[[WebDriver,float],bool]    = None
        self.current_select_element:    WebElement                          = None
    def __del__(self):
        self.__browser.close()

    def get_browser(self) -> WebDriver:
        '''
        get basic browser driver
        '''
        return self.__browser

    #set or get attributes about time-wait
    @property
    def delay(self) -> float:
        return self.__delay
    @delay.setter
    def delay(self, delay:float):
        self.__delay = delay
    def set_implicitly_wait_mode(self):
        self.wait_enable = implicitly_wait_enable_constexpr_value
        return self
    def set_if_wait_mode(self):
        self.wait_enable = if_wait_enable_constexpr_value
        return self
    def set_no_wait_mode(self):
        self.wait_enable = no_wait_enable_constexpr_value
        return self
    def wait_delay(self) -> Self:
        if self.delay_notify is False:
            return self
        elif self.wait_enable == implicitly_wait_enable_constexpr_value:
            self.__browser.implicitly_wait(self.__delay)
        elif self.wait_enable == if_wait_enable_constexpr_value:
            self.__if_wait_pred(self.__browser,self.__delay)
        else:
            time.sleep(self.delay)
        return self

    #function pred is bool(WebDriver,delay:float)
    #can catch from EC
    def set_if_wait_for(self, pred:Callable[[WebDriver, float], bool]):
        self.__if_wait_pred = pred
        return self

    #open url
    def open_url(self, url):
        self.__browser.get(url)
        return self.wait_delay()

    #window stats but not update window currently
    def set_scale(self, x:int, y:int):
        self.__transform.scale[0] = x
        self.__transform.scale[1] = y
        return self
    def set_rect(self, width:int, height:int):
        self.__transform.scale(width, height)
        return self
    def set_position(self, x:int, y:int):
        self.__transform.move(x-self.__transform.get_left_edge(), y-self.__transform.get_bottom_edge())
        return self
    def move(self, dx:int, dy:int):
        self.__transform.move(dx, dy)
        return self
    #update window stats now
    def update_transform(self):
        pos:        Tuple[NumberLike, NumberLike] = self.__transform.get_lb_pos()
        datasize:   Tuple[NumberLike, NumberLike] = self.__transform.get_width(), self.__transform.get_height()
        self.__browser.set_window_position(pos[0], pos[1])
        self.__browser.set_window_size(datasize[0], datasize[1])
        return self.wait_delay()
    def full_window(self):
        self.__browser.fullscreen_window()
        return self.wait_delay()

    #toolkit
    def search(
        self,
        target:     str,
        url:        str        = None,
        element:    WebElement = None
        ) -> Self:
        if url is not None:
            self.open_url(url)
        if element is None:
            element = self.__browser.find_element(By.ID, 'sb_form_q')
        if element is None :
            self.__browser.get("https://www.bing.com")
            element = self.__browser.find_element(By.ID, 'sb_form_q')
        element.send_keys(target)
        element.submit()
        return self.wait_delay()
    def switch_to_frame(
        self,
        frame_ref:Optional[Union[str, int, WebElement]] = None
        ) -> Self:
        self.__browser.switch_to.frame(self.current_select_element if frame_ref is None else frame_ref)
        return self.wait_delay()
    def switch_to_parent(self) -> Self:
        self.__browser.switch_to.parent_frame()
        return self.wait_delay()
    def alert_accept(self) -> Self:
        self.__browser.switch_to.alert.accept()
        return self.wait_delay()
    def alert_dismiss(self) -> Self:
        self.__browser.switch_to.alert.dismiss()
        return self.wait_delay()
    def screenshot(self, path:Optional[Union[str, tool_file]] = None):
        self.__browser.get_screenshot_as_file("./screenshot.png" if path is None else Unwrapper2Str(path))
        return self.wait_delay()

    #seek for element
    def find_element(
            self,
            name:str,
            typen:str) -> WebElement:
        return self.__browser.find_element(typen,name)
    def find_elements(
            self,
            name:str,
            typen:ByType) -> List[WebElement]:
        return self.__browser.find_elements(typen,name)
    def find_class(self,name:str) -> WebElement:
        return self.find_element(name,By.CLASS_NAME)
    def find_xpath(self,name:str) -> WebElement:
        return self.find_element(name,By.XPATH)
    def find_link_text(self,name:str) -> WebElement:
        return self.find_element(name,By.LINK_TEXT)
    def find_partial_link_text(self,name:str) -> WebElement:
        return self.find_element(name,By.PARTIAL_LINK_TEXT)
    def find_tag(self,name:str) -> WebElement:
        return self.find_element(name,By.TAG_NAME)
    def find_css(self,name:str) -> WebElement:
        return self.find_element(name,By.CSS_SELECTOR)
    def find_id(self,name:str) -> WebElement:
        return self.find_element(name,By.ID)
    def find_class_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.CLASS_NAME)
    def find_xpath_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.XPATH)
    def find_link_text_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.LINK_TEXT)
    def find_Partial_link_text_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.PARTIAL_LINK_TEXT)
    def find_tag_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.TAG_NAME)
    def find_css_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.CSS_SELECTOR)
    def find_id_s(self,name:str) -> WebElement:
        return self.find_elements(name,By.ID)
    def is_display(self, element:WebElement) -> bool:
        return element.is_displayed()
    def is_enable(self, element:WebElement) -> bool:
        return element.is_enabled()
    def is_select(self, element:WebElement) -> bool:
        return element.is_selected()

    #current page operator
    def refresh(self):
        self.__browser.refresh()
        return self.wait_delay()
    def forward(self):
        self.__browser.forward()
        return self.wait_delay()
    def backward(self):
        self.__browser.back()
        return self.wait_delay()

    #get infomations
    def get_title(self) -> str:
        return self.__browser.title
    @property
    def title(self):
        return self.get_title()
    @title.setter
    def title(self, title:str):
        self.__browser.title = title
        return title
    def get_url(self) -> str:
        return self.__browser.current_url
    def get_browser_name(self) -> str:
        return self.__browser.name
    def get_source(self):
        return self.__browser.page_source

    #mouse actions
    def get_mouse_action(self) -> Mouse:
        if self.__mouse_action is None:
            self.__mouse_action = Mouse(self.__browser)
        return self.__mouse_action
    def click(self,element:WebElement):
        self.get_mouse_action().click(element)
        return self.wait_delay()
    def right_click(self,element:WebElement):
        self.get_mouse_action().context_click(element)
        return self.wait_delay()
    def double_click(self,element:WebElement):
        self.get_mouse_action().double_click(element)
        return self.wait_delay()
    def drag_and_drop(self,element:WebElement, offset):
        self.get_mouse_action().drag_and_drop_by_offset(element,offset[0],offset[1])
        return self.wait_delay()
    def move_to(self,element:WebElement,offset):
        self.get_mouse_action().move_to_element_with_offset(element,offset[0],offset[1])
        return self.wait_delay()
    def perfrom(self):
        self.get_mouse_action().perform()
        return self.wait_delay()

    #select target element and set it current select
    def select_element(self, element:WebElement):
        self.current_select_element = element
        return self

    #send keys to target element
    def send_keys_to_element(self, element:WebElement, keys:str):
        element.send_keys(keys)
        return self.wait_delay()
    def send_keys(self, keys:str):
        return self.send_keys_to_element(self.current_select_element, keys)
    def send_keys_to_elements(self, elements:List[WebElement], keys:str):
        for element in elements:
            element.send_keys(keys)
        return self.wait_delay()

    #actions about cookie
    def get_cookie(self, name) -> Dict:
        return self.__browser.get_cookie(name)
    def get_cookies(self) -> List[dict]:
        return self.__browser.get_cookies()
    def add_cookie(self, cookie_name, cookie_value):
        self.__browser.add_cookie({cookie_name,cookie_value})
        return self
    def add_cookies(self, cookie_dicts):
        self.__browser.add_cookie(cookie_dicts)
        return self
    def erase_cookie(self,name):
        self.__browser.delete_cookie(name)
        return self
    def clear_cookies(self):
        self.__browser.delete_all_cookies();
        return self

    #toolkits
    def is_element_exist(self, name:str) -> bool:
        try:
            self.find_xpath_s(name)
            return True
        except:
            return False
    def run_js(self, js:str):
        self.__browser.execute_script(js)
        return self.wait_delay()
    def run_aynsc_js(self, js:str):
        self.__browser.execute_async_script(js)
        return self.wait_delay()

    #create if-wait task
    def create_wait_task(self, timeout:int = -1, poll_freq:int = 0.5, ignored_exceptions: Optional[Iterable[Type[Exception]]] = None) -> WaitTask:
        return WaitTask(self.__browser, self.get_delay()if timeout<=0 else timeout, poll_freq, ignored_exceptions)
    DriverOrElement = TypeVar("DriverOrElement", bound=Union[WebDriver, WebElement])
    WaitTaskT = TypeVar("WaitTaskT")
    def wait_until(self, task:WaitTask, method: Callable[[DriverOrElement], Union[Literal[False], WaitTaskT]], message: str = "") -> WaitTaskT:
        return task.until(method,message)
    def until_find_element(self, task:WaitTask, name:str, typen:str = By.ID) -> WaitTaskT:
        return self.wait_until(task,EC.presence_of_element_located(typen,name))
    def wait_until_not(self, task:WaitTask, method: Callable[[DriverOrElement], WaitTaskT], message: str = "") -> Union[WaitTaskT, Literal[True]]:
        return task.until_not(method,message)
    def until_not_find_element(self, task:WaitTask, name:str, typen:str = By.ID) -> Union[WaitTaskT, Literal[True]]:
        return self.wait_until_not(task,EC.presence_of_element_located(typen,name))

class test_selunit(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        self.__unit:selunit = None
        super().__init__(methodName)

    def get_unit(self) -> selunit:
        if self.__unit is None:
            self.__unit = selunit()
        return self.__unit
    def set_unit(self, unit:selunit) -> None:
        self.__unit = unit
    def get_browser(self) -> WebDriver:
        return self.get_unit().get_browser()
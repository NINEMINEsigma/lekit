from typing                                                 import *
from lekit.MathEx.Transform                                 import *
from lekit.File.Core                                        import tool_file, UnWrapper as Unwrapper2Str, is_loss_tool_file
from selenium.webdriver.remote.webdriver                    import WebDriver
from selenium.webdriver.remote.webelement                   import WebElement
from selenium.webdriver.common.by                           import By, ByType
from selenium.webdriver.common.action_chains                import ActionChains as Mouse
from selenium.webdriver.support.ui import WebDriverWait     as     WaitTask
from selenium.webdriver.support import expected_conditions  as     EC
import                                                             time
import                                                             unittest
import                                                             tqdm

import selenium.webdriver                                   as     base_webdriver
from selenium.types                                         import *
import selenium.common                                      as     base_common

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
type ByTypen = str
type DriverOrElement = Union[WebDriver, WebElement]

def make_xpath_contains(element:str, value:str) -> str:
    return f"(contains({element}, '{value}'))"
def make_xpath_contains_s(*args: Tuple[str, str]) -> str:
    cur = [make_xpath_contains(key, value) for key, value in args]
    return "("+" or ".join(cur)+")"

class basic_unit_interface[_T:DriverOrElement](left_value_reference[_T]):
    __inject_browser: WebDriver = None

    @property
    def browser(self) -> WebDriver:
        return self.__inject_browser
    @browser.setter
    def browser(self, value:WebDriver) -> None:
        self.__inject_browser = value

    def get_browser(self) -> WebDriver:
        '''
        get basic browser driver
        '''
        return self.browser

    def __init__(self, ref_value:_T, browser:Union[
        left_value_reference[WebDriver],
        WebDriver
    ]):
        super().__init__(ref_value)
        self.browser = UnwrapperInstance2Ref(browser)
        if browser is None:
            raise ValueError("browser is None")
        self.__inject_browser = browser

    #seek for element
    def find_element(
            self,
            name:str,
            typen:ByTypen) -> Self[WebElement]:
        return basic_unit_interface[WebElement](
            self.ref_value.find_element(typen,name)
        )
    def find_elements(
            self,
            name:str,
            typen:ByType) -> List[Self[WebElement]]:
        return [
            basic_unit_interface[WebElement](element)
            for element
            in self.ref_value.find_elements(typen,name)
            ]
    def find_name(self,name:str):
        return self.find_element(name,By.NAME)
    def find_name_s(self,name:str):
        return self.find_elements(name,By.NAME)
    def find_class(self,name:str):
        return self.find_element(name,By.CLASS_NAME)
    def find_xpath(self,name:str):
        return self.find_element(name,By.XPATH)
    def find_link_text(self,name:str):
        return self.find_element(name,By.LINK_TEXT)
    def find_partial_link_text(self,name:str) -> WebElement:
        return self.find_element(name,By.PARTIAL_LINK_TEXT)
    def find_tag(self,name:str) -> WebElement:
        return self.find_element(name,By.TAG_NAME)
    def find_css(self,name:str) -> WebElement:
        return self.find_element(name,By.CSS_SELECTOR)
    def find_id(self,name:str) -> WebElement:
        return self.find_element(name,By.ID)
    def find_class_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.CLASS_NAME)
    def find_xpath_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.XPATH)
    def find_link_text_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.LINK_TEXT)
    def find_partial_link_text_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.PARTIAL_LINK_TEXT)
    def find_tag_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.TAG_NAME)
    def find_css_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.CSS_SELECTOR)
    def find_id_s(self,name:str) -> List[WebElement]:
        return self.find_elements(name,By.ID)
    def is_display(self, element:WebElement) -> bool:
        if element is None:
            return False
        return element.is_displayed()
    def is_enable(self, element:WebElement) -> bool:
        if element is None:
            return False
        return element.is_enabled()
    def is_select(self, element:WebElement) -> bool:
        if element is None:
            return False
        return element.is_selected()

    def send_keys(self, messages:str):
        return self.ref_value.send_keys(messages)

class selunit(basic_unit_interface[WebDriver]):
    """ instance of selenium functions """

    #web browser driver(default is MSedge)
    # "" __browser ""
    def __init__(
            self,
            browser:    WebDriver   = None,
            transform:  abs_box     = Rect(int(50), int(50), int(900), int(800)),
            delay = 3):
        if browser is None:
            browser = base_webdriver.Edge()
        super().__init__(browser, browser)
        self.__transform:               abs_box                             = transform
        self.delay:                     float                               = delay
        self.delay_notify:              float                               = True
        self.wait_enable:               wait_enable_type                    = no_wait_enable_type
        self.__if_wait_pred:            Callable[[WebDriver,float],bool]    = None
        self.current_select_element:    WebElement                          = None
    def __del__(self):
        self.browser.close()
    @override
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.quit()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def quit(self) -> Self:
        self.browser.quit()
        return self

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
            self.ref_value.implicitly_wait(self.__delay)
        elif self.wait_enable == if_wait_enable_constexpr_value:
            self.__if_wait_pred(self.ref_value,self.__delay)
        else:
            time.sleep(self.delay)
        return self
    def wait_without_notify(self, value:float) -> Self:
        time.sleep(value)
        return self

    # toolkit of wait
    def make_wait_task(
            self,
            timeout:    float = 10,
            ) -> WaitTask:
        return WaitTask(self.browser, timeout)

    #function pred is bool(WebDriver,delay:float)
    #can catch from EC
    def set_if_wait_for(self, pred:Callable[[WebDriver, float], bool]):
        self.__if_wait_pred = pred
        return self

    #open url
    def inject_open_url(self, url):
        self.ref_value.get(url)
        return self.wait_delay()
    def open(self, url):
        return self.inject_open_url(url)

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
        self.ref_value.set_window_position(pos[0], pos[1])
        self.ref_value.set_window_size(datasize[0], datasize[1])
        return self.wait_delay()
    def full_window(self):
        self.ref_value.fullscreen_window()
        return self.wait_delay()

    #toolkit
    def search(
        self,
        target:     str,
        url:        str        = None,
        element:    WebElement = None
        ) -> Self:
        if url is not None:
            self.inject_open_url(url)
        if element is None:
            element = self.ref_value.find_element(By.ID, 'sb_form_q')
        if element is None :
            self.ref_value.get("https://www.bing.com")
            element = self.ref_value.find_element(By.ID, 'sb_form_q')
        element.send_keys(target)
        element.submit()
        return self.wait_delay()
    def switch_to_frame(
        self,
        frame_ref:Optional[Union[str, int, WebElement]] = None
        ) -> Self:
        self.ref_value.switch_to.frame(self.current_select_element if frame_ref is None else frame_ref)
        return self.wait_delay()
    def switch_to_parent(self) -> Self:
        self.ref_value.switch_to.parent_frame()
        return self.wait_delay()
    def switch_to_page(self, index:int) -> Self:
        self.browser.switch_to.window(self.ref_value.window_handles[index])
        return self.wait_delay()
    def alert_accept(self) -> Self:
        self.ref_value.switch_to.alert.accept()
        return self.wait_delay()
    def alert_dismiss(self) -> Self:
        self.ref_value.switch_to.alert.dismiss()
        return self.wait_delay()
    def screenshot(self, path:Union[str, tool_file] = "./screenshot.png") -> Self:
        if is_loss_tool_file(path) is False:
            self.ref_value.get_screenshot_as_file(Unwrapper2Str(path))
            return self.wait_delay()
        else:
            return self

    # toolkit of find
    def find_password(self) -> List[WebElement]:
        result = self.find_xpath_s("//*[(@password or @pwd)]")
        result.extend(
            self.find_xpath_s(f"//*[{make_xpath_contains_s(
                    ("@*", "password"),
                    ("@*", "pwd")
            )}]")
        )
        return result
    def find_username(self) -> List[WebElement]:
        result = self.find_xpath_s("//*[(@username or @user or @email or @phone or @tel)]")
        result.extend(
            self.find_xpath_s(
                f"//*[{make_xpath_contains_s(
                    ("@*", "username"),
                    ("@*", "user"),
                    ("@*", "email"),
                    ("@*", "phone"),
                    ("@*", "tel")
                )}]")
            )
        return result
    def find_text(self) -> List[WebElement]:
        result = self.find_xpath_s("//*[(@text or @value or @placeholder or @title)]")
        result.extend(
            self.find_xpath_s(f"//*[{make_xpath_contains_s(
                    ("@*", "text"),
                    ("@*", "value"),
                    ("@*", "placeholder"),
                    ("@*", "title")
            )}]")
        )
        return result
    def login(self) -> bool:
        result = self.find_xpath_s("//*[(@login or @signIn or @confirm)]")
        result.extend(
            self.find_xpath_s(f"//*[{make_xpath_contains_s(
                    ("@*", "login"),
                    ("@*", "signIn"),
                    ("@*", "confirm")
            )}]")
        )
        if len(result) != 0:
            result[0].click()
            return True
        return False
    def find_anylike_with_xpath(self, name:str) -> List[WebElement]:
        result = self.find_xpath_s(f"//*[@{name}]")
        result.extend(
            self.find_xpath_s(f"//*[contains(@{name})]")
        )
        return result

    #current page operator
    def refresh(self):
        self.ref_value.refresh()
        return self.wait_delay()
    def forward(self):
        self.ref_value.forward()
        return self.wait_delay()
    def backward(self):
        self.ref_value.back()
        return self.wait_delay()
    def window_handles(self) -> List[str]:
        return self.ref_value.window_handles

    #get infomations
    def get_title(self) -> str:
        return self.ref_value.title
    @property
    def title(self):
        return self.get_title()
    @title.setter
    def title(self, title:str):
        self.ref_value.title = title
        return title
    def get_url(self) -> str:
        return self.ref_value.current_url
    def get_browser_name(self) -> str:
        return self.ref_value.name
    def get_source(self):
        return self.ref_value.page_source

    #mouse actions
    def get_mouse_action(self) -> Mouse:
        if self.__mouse_action is None:
            self.__mouse_action = Mouse(self.ref_value)
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
    def send_keys_to_elements(self, elements:List[WebElement], keys:str):
        for element in elements:
            element.send_keys(keys)
        return self.wait_delay()

    #actions about cookie
    def get_cookie(self, name) -> Dict:
        return self.ref_value.get_cookie(name)
    def get_cookies(self) -> List[dict]:
        return self.ref_value.get_cookies()
    def add_cookie(self, cookie_name, cookie_value):
        self.ref_value.add_cookie({cookie_name,cookie_value})
        return self
    def add_cookies(self, cookie_dicts):
        self.ref_value.add_cookie(cookie_dicts)
        return self
    def erase_cookie(self,name):
        self.ref_value.delete_cookie(name)
        return self
    def clear_cookies(self):
        self.ref_value.delete_all_cookies()
        return self

    #toolkits
    def is_element_exist(self, name:str) -> bool:
        try:
            self.find_xpath_s(name)
            return True
        except:
            return False
    def run_js(self, js:str):
        self.ref_value.execute_script(js)
        return self.wait_delay()
    def run_aynsc_js(self, js:str):
        self.ref_value.execute_async_script(js)
        return self.wait_delay()

    #create if-wait task
    def create_wait_task(
            self,
            timeout:            int = -1,
            poll_freq:          int = 0.5,
            ignored_exceptions: Optional[Iterable[Type[Exception]]] = None
            ) -> WaitTask:
        return WaitTask(self.ref_value, self.delay if timeout<=0 else timeout, poll_freq, ignored_exceptions)
    def wait_until[_Element_or_False:Union[Literal[False], DriverOrElement]](
            self,
            task:               WaitTask,
            method:             Callable[[DriverOrElement], _Element_or_False],
            message:            str = ""
            ) -> _Element_or_False:
        return task.until(method, message)
    def until_find_element(
            self,
            task:               WaitTask,
            name:               str,
            typen:              str = By.ID
            ) -> WebElement:
        return self.wait_until(task, EC.presence_of_element_located((typen,name)))
    def wait_until_not[_Element_or_True:Union[Literal[True], DriverOrElement]](
            self,
            task:               WaitTask,
            method:             Callable[[DriverOrElement], _Element_or_True],
            message:            str = ""
            ) -> _Element_or_True:
        return task.until_not(method, message)
    def until_not_find_element(
            self,
            task:               WaitTask,
            name:               str,
            typen:              str = By.ID
            ) -> WebElement:
        return self.wait_until_not(task, EC.presence_of_element_located((typen,name)))

class test_selunit(left_value_reference[selunit], unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        left_value_reference[selunit].__init__(self, None)
        unittest.TestCase.__init__(self, methodName)
    @property
    def browser(self) -> selunit:
        return self.ref_value
    def setUp(self) -> None:
        self.browser = selunit()
    def tearDown(self) -> None:
        self.browser.quit()

selunit_instance:WebDriver = None
__internal_build_up:Callable[[], selunit] = lambda: selunit()
def set_buildup_static_selunit_instance_func(func:Callable[[], selunit]):
    global __internal_build_up
    __internal_build_up = func

class page_interface(left_value_reference[selunit], invoke_callable, ABC):
    def __init__(self, ref_value):
        super().__init__(ref_value)

    @abstractmethod
    def next_page(self) -> Self:
        raise NotImplementedError("next_page is not implemented.")

class page(page_interface, ActionEvent[Action[selunit]], ABC):
    def __init__(self, *actions:Action[selunit]) -> None:
        if selunit_instance is None:
           selunit_instance = __internal_build_up()
        page_interface.__init__(self, selunit_instance)
        ActionEvent[Action[selunit]].__init__(self, actions)

    @abstractmethod
    def inject_next_page(self) -> page_interface:
        """Injects the next page into the current page."""
        raise NotImplementedError("inject_next_page is not implemented.")

    @override
    def next_page(self) -> page_interface:
        self.invoke()
        return self.inject_next_page()

    @override
    def _inject_invoke(self):
        result:List[Any] = []
        for index in self.call_indexs:
            try:
                result.append(self.__actions[index]())
            except Exception as ex:
                result.append(ex)
        return result
    def invoke(self) -> Self:
        print(f"{self.SymbolName()} is invoking...")
        for index in tqdm.tqdm(range(len(self.actions))):
            self.actions[index](self.ref_value)







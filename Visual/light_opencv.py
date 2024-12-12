import cv2 as base

class light_cv_camera:
    def __init__(self, index:int = 0):
        self.capture = base.VideoCapture(index)
        
    def is_open(self):
        return self.capture.isOpened()
        
    def release(self):
        self.capture.release()
    def retarget(self, index:int = 0):
        self.capture.release()
        self.capture = base.VideoCapture(index)
        return self
        
    def current_frame(self):
        _, frame = self.capture.read()
        return frame
    def current_stats(self):
        stats,_ = self.capture.read()
        return stats
    
    def save_current_frame(self, file_name:str = "current.png"):
        base.imwrite(file_name, self.current_frame())
        return self
    def show_current_frame(self, window_name:str = 'frame'):
        base.imshow(window_name, self.current_frame())
        base.waitKey(0)
        return self
    
    def get_captrue_info(self, id:int):
        return self.capture.get(id)
    def get_prop_pos_msec(self):
        return self.get_captrue_info(0)
    def get_prop_pos_frames(self):
        return self.get_captrue_info(1)
    def get_prop_avi_ratio(self):
        return self.get_captrue_info(2)
    def get_prop_frame_width(self):
        return self.get_captrue_info(3)
    def get_prop_frame_height(self):
        return self.get_captrue_info(4)
    def get_prop_fps(self):
        return self.get_captrue_info(5)
    def get_prop_fourcc(self):
        return self.get_captrue_info(6)
    def get_prop_frame_count(self):
        return self.get_captrue_info(7)
    def get_prop_format(self):
        return self.get_captrue_info(8)
    def get_prop_mode(self):
        return self.get_captrue_info(9)
    def get_prop_brightness(self):
        return self.get_captrue_info(10)
    def get_prop_contrast(self):
        return self.get_captrue_info(11)
    def get_prop_saturation(self):
        return self.get_captrue_info(12)
    def get_prop_hue(self):
        return self.get_captrue_info(13)
    def get_prop_gain(self):
        return self.get_captrue_info(14)
    def get_prop_exposure(self):
        return self.get_captrue_info(15)
    def get_prop_convert_rgb(self):
        return self.get_captrue_info(16)
        
    def setup_capture(self, id:int, value):
        self.capture.set(id, value)
        return self
    def set_prop_pos_msec(self, value:int):
        return self.setup_capture(0, value)
    def set_prop_pos_frames(self, value:int):
        return self.setup_capture(1, value)
    def set_prop_avi_ratio(self, value:float):
        return self.setup_capture(2, value)
    def set_prop_frame_width(self, value:int):
        return self.setup_capture(3, value)
    def set_prop_frame_height(self, value:int):
        return self.setup_capture(4, value)
    def set_prop_fps(self, value:int):
        return self.setup_capture(5, value)
    def set_prop_fourcc(self, value):
        return self.setup_capture(6, value)
    def set_prop_frame_count(self, value):
        return self.setup_capture(7, value)
    def set_prop_format(self, value):
        return self.setup_capture(8, value)
    def set_prop_mode(self, value):
        return self.setup_capture(9, value)
    def set_prop_brightness(self, value):
        return self.setup_capture(10, value)
    def set_prop_contrast(self, value):
        return self.setup_capture(11, value)
    def set_prop_saturation(self, value):
        return self.setup_capture(12, value)
    def set_prop_hue(self, value):
        return self.setup_capture(13, value)
    def set_prop_gain(self, value):
        return self.setup_capture(14, value)
    def set_prop_exposure(self, value):
        return self.setup_capture(15, value)
    def set_prop_convert_rgb(self, value:int):
        return self.setup_capture(16, value)
    def set_prop_rectification(self, value:int):
        return self.setup_capture(17, value)
    
    def set_window_rect(self, weight:int, height:int):
        self.set_prop_frame_width(weight)
        self.set_prop_frame_height(height)
        return self
    def get_window_rect(self):
        return self.get_prop_frame_width(), self.get_prop_frame_height()
    
    def set_window_size(self, weight:int, height:int):
        return self.set_window_rect(weight, height)
    def set_window_name(self, name:str):
        base.namedWindow(name)
        return self

    def get_frame(self):
        _, frame = self.capture.read()
        return frame
    def show_frame(self, quit_key:str, wait_key:int=1, name:str='frame'):
        while True:
            frame = self.get_frame()
            base.imshow(name, frame)
            if base.waitKey(wait_key) & 0xFF == ord(quit_key):
                break

    def show_gray(self):
        while True:
            _, frame = self.capture.read()
            gray = base.cvtColor(frame, base.COLOR_BGR2GRAY)
            base.imshow('gray', gray)
            if base.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        base.destroyAllWindows()

    def detect_faces(self):
        face_cascade = base.CascadeClassifier('haarcascade_frontalface_default.xml')
        while True:
            _, frame = self.capture.read()
            gray = base.cvtColor(frame, base.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                base.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            base.imshow('img',frame)
            if base.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        base.destroyAllWindows()

    def apply_cascade(self, cascade_path):
        cascade = base.CascadeClassifier(cascade_path)
        while True:
            _, frame = self.capture.read()
            gray = base.cvtColor(frame, base.COLOR_BGR2GRAY)
            objects = cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in objects:
                base.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            base.imshow('img',frame)
            if base.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        base.destroyAllWindows()

class light_cv_window:
    def __init__(self, name:str):
        self.__my_window_name = name
        base.namedWindow(name)
    def __del__(self):
        base.destroyWindow(self.__my_window_name)

    def show_image(self, image):
        base.imshow(self.__my_window_name, image)
    def destroy(self):
        if self.__my_window_name is not None:
            base.destroyWindow(self.__my_window_name)
            self.__my_window_name=None
            
    def set_window_size(self, weight:int, height:int):
        base.resizeWindow(self.__my_window_name, weight, height)
        return self
    def get_window_size(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_WIDTH), base.getWindowProperty(self.__my_window_name, base.WINDOW_HEIGHT)
    
    def get_window_property(self, prop_id:int):
        return base.getWindowProperty(self.__my_window_name, prop_id)
    def set_window_property(self, prop_id:int, prop_value:int):
        return base.setWindowProperty(self.__my_window_name, prop_id, prop_value)
    def get_prop_frame_width(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_WIDTH)
    def get_prop_frame_height(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_HEIGHT)
    def is_full_window(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN) > 0
    def set_full_window(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN, 1)
    def set_normal_window(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN, 0)
    def is_using_openGL(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_OPENGL) > 0
    def set_using_openGL(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_OPENGL, 1)
    def set_not_using_openGL(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_OPENGL, 0)
    def is_autosize(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE) > 0
    def set_autosize(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE, 1)
    def set_not_autosize(self):
        return base.setWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE, 0)
    
    def set_window_rect(self, x:int, y:int, weight:int, height:int):
        base.moveWindow(self.__my_window_name, x, y)
        return self.set_window_size(weight, height)

    def set_window_pos(self, x:int, y:int):
        base.moveWindow(self.__my_window_name, x, y)
        return self

    def wait_key(self, wait_time:int=0):
        return base.waitKey(wait_time)


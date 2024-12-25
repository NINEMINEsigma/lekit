from typing             import *

import cv2              as     base
import cv2.data         as     BaseData
from cv2.typing         import *
import numpy            as     np
from PIL                import ImageFile, Image

from lekit.Str.Core     import UnWrapper as Unwrapper2Str
from lekit.File.Core    import tool_file, Wrapper as Wrapper2File

from lekit.Lang.BaseClass import BaseBehaviour

VideoWriter = base.VideoWriter
def mp4_with_MPEG4_fourcc() -> int:
    return VideoWriter.fourcc(*"mp4v")
def avi_with_Xvid_fourcc() -> int: 
    return VideoWriter.fourcc(*"XVID")
def avi_with_DivX_fourcc() -> int:
    return VideoWriter.fourcc(*"DIVX")
def avi_with_MJPG_fourcc() -> int:
    return VideoWriter.fourcc(*"MJPG")
def mp4_or_avi_with_H264_fourcc() -> int:
    return VideoWriter.fourcc(*"X264")
def avi_with_H265_fourcc() -> int:
    return VideoWriter.fourcc(*"H264")
def wmv_with_WMV1_fourcc() -> int:
    return VideoWriter.fourcc(*"WMV1")
def wmv_with_WMV2_fourcc() -> int:
    return VideoWriter.fourcc(*"WMV2")
def oggTheora_with_THEO_fourcc() -> int:
    return VideoWriter.fourcc(*"THEO")
def flv_with_FLV1_fourcc() -> int:
    return VideoWriter.fourcc(*"FLV1")
class VideoWriterInstance(VideoWriter):
    def __init__(
        self, 
        file_name:  Union[tool_file, str], 
        fourcc:     int,
        fps:        float, 
        frame_size: tuple[int, int],
        is_color:   bool = True
        ):
        super().__init__(Unwrapper2Str(file_name), fourcc, fps, frame_size, is_color)
    def __del__(self):
        self.release()

def wait_key(delay:int):
    return base.waitKey(delay)
def until_esc():
    return wait_key(0)

def is_current_key(key:str, *, wait_delay:int = 1):
    return wait_key(wait_delay) & 0xFF == ord(key[0])

class light_cv_view:
    def __init__(self, filename_or_index:Union[str, tool_file, int]):
        self.__capture: base.VideoCapture   = None
        self.stats:     bool                = True
        self.retarget(filename_or_index)
    def __del__(self):
        self.release()
    
    def __bool__(self):
        return self.stats
    
    def is_open(self):
        return self.__capture.isOpened()
        
    def release(self):
        if self.__capture is not None:
            self.__capture.release()
    def retarget(self, filename_or_index:Union[str, tool_file, int]):
        self.release()
        if isinstance(filename_or_index, int):
            self.__capture = base.VideoCapture(filename_or_index)
        else:
            self.__capture = base.VideoCapture(Unwrapper2Str(filename_or_index))
        return self
    
    def next_frame(self) -> MatLike:
        self.stats, frame =self.__capture.read()
        if self.stats:
            return frame
        else:
            return None
    
    def get_captrue_info(self, id:int):
        return self.__capture.get(id)
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
        self.__capture.set(id, value)
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
    
    @property
    def frame_size(self) -> Tuple[float, float]:
        return self.get_prop_frame_width(), self.get_prop_frame_height()
    
class light_cv_camera(light_cv_view):
    def __init__(self, index:int = 0):
        self.writer:    VideoWriter = None
        super().__init__(int(index))
    
    @override
    def release(self):
        super().release()
        if self.writer is not None:
            self.writer.release()
    
    def current_frame(self):
        return self.next_frame()
    
    def recording(
        self, 
        stop_pr:    Callable[[], bool], 
        writer:     VideoWriter,
        ):
        self.writer = writer
        while self.is_open():
            if stop_pr():
                break
            frame = self.current_frame()
            base.imshow("__recording__", frame)
            writer.write(frame)
        base.destroyWindow("__recording__")
        return self

class ImageObject:
    def __init__(
        self,
        image:          Optional[Union[
            str,
            Self,
            light_cv_camera,
            tool_file, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]],
        flags:          int             = -1):
        self.__image:   MatLike         = None
        self.__camera:  light_cv_camera = None
        self.current:   MatLike         = None
        if isinstance(image, light_cv_camera):
            self.lock_from_camera(image)
        else:
            self.load_image(image, flags)

    @property
    def camera(self) -> light_cv_camera:
        if self.__camera is None or self.__camera.is_open() is False:
            return None
        else:
            return self.__camera
    @property
    def image(self) -> MatLike:
        if self.current is not None:
            return self.current
        elif self.camera is None:
            return self.__image
        else:
            return self.__camera.current_frame()

    @image.setter
    def image(self, image:          Optional[Union[
            str,
            Self,
            tool_file, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]):
        self.load_image(image)

    def load_from_nparray(
        self,
        array_: np.ndarray,
        code:   int = base.COLOR_RGB2BGR,
        *args, **kwargs
        ):
        self.__image = base.cvtColor(array_, code, *args, **kwargs)
        return self
    def load_from_PIL_image(
        self,
        image:  Image.Image,
        code:   int = base.COLOR_RGB2BGR,
        *args, **kwargs
    ):
        self.load_from_nparray(np.array(image), code, *args, **kwargs)
        return self
    def load_from_PIL_ImageFile(
        self,
        image:  ImageFile.ImageFile,
        rect:   Optional[Tuple[float, float, float, float]] = None
    ):
        return self.load_from_PIL_image(image.crop(rect))
    def load_from_cv2_image(self, image:  MatLike):
        self.__image = image
        return self
    def lock_from_camera(self, camera: light_cv_camera):
        self.__camera = camera
        return self

    @property
    def dimension(self) -> int:
        return self.image.ndim
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        '''height, width, depth'''
        return self.image.shape
    @property
    def height(self) -> int:
        return self.shape[0]
    @property
    def width(self) -> int:
        return self.shape[1]

    def is_enable(self):
        return self.image is not None
    def is_invalid(self):
        return self.is_enable() is False
    def __bool__(self):
        return self.is_enable()
    def __MatLike__(self):
        return self.image

    def load_image(
        self, 
        image:          Optional[Union[
            str,
            tool_file, 
            Self,
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]],
        flags:          int = -1
        ):
        """加载图片"""
        if image is None:
            self.__image = None
            return self
        elif isinstance(image, type(self)):
            self.__image = image.image
        elif isinstance(image, MatLike):
            self.__image = image
        elif isinstance(image, np.ndarray):
            self.load_from_nparray(image, flags)
        elif isinstance(image, ImageFile.ImageFile):
            self.load_from_PIL_ImageFile(image, flags)
        elif isinstance(image, Image.Image):
            self.load_from_PIL_image(image, flags)
        else:
            self.__image = base.imread(Unwrapper2Str(image), flags)
        return self
    def save_image(self, save_path:Union[str, tool_file], is_path_must_exist = False):
        """保存图片"""
        if is_path_must_exist:
            Wrapper2File(save_path).try_create_parent_path()
        if self.is_enable():
            base.imwrite(Unwrapper2Str(save_path), self.image)
        return self

    def show_image(
        self, 
        window_name:        str                         = "Image", 
        delay:              Union[int,str]              = 0,
        image_show_func:    Callable[[Self], None]      = None,
        *args, **kwargs
        ):
        """显示图片"""
        if self.is_invalid():
            return self
        if self.camera is not None:
            while wait_key(1) & 0xFF != ord(str(delay)[0]) and self.camera is not None:
                # dont delete this line, self.image is camera flame now, see<self.current = None>
                self.current = self.image
                if image_show_func is not None:
                    image_show_func(self)
                if self.current is not None:
                    base.imshow(window_name, self.current)
                # dont delete this line, see property<image>
                self.current = None
        else:
            base.imshow(window_name, self.image)
            base.waitKey(delay = int(delay), *args, **kwargs)
        if base.getWindowProperty(window_name, base.WND_PROP_VISIBLE) > 0:
            base.destroyWindow(window_name)
        return self

    # 分离通道
    def split(self):
        """分离通道"""
        return base.split(self.image)
    def split_to_image_object(self):
        """分离通道"""
        return [ImageObject(channel) for channel in self.split()]
    @property
    def channels(self):
        return self.split()
    @property
    def blue_channel(self):
        return self.channels[0]
    @property
    def green_channel(self):
        return self.channels[1]
    @property
    def red_channel(self):
        return self.channels[2]
    @property
    def alpha_channel(self):
        return self.channels[3]
    def get_blue_image(self):
        return ImageObject(self.blue_channel)
    def get_green_image(self):
        return ImageObject(self.green_channel)
    def get_red_image(self):
        return ImageObject(self.red_channel)
    def get_alpha_image(self):
        return ImageObject(self.alpha_channel)

    # 混合通道
    def merge_channels_from_list(self, channels:List[MatLike]):
        """合并通道"""
        self.image = base.merge(channels)
        return self
    def merge_channels(self, blue:MatLike, green:MatLike, red:MatLike):
        """合并通道"""
        return self.merge_channels_from_list([blue, green, red])
    def merge_channel_list(self, bgr:List[MatLike]):
        """合并通道"""
        return self.merge_channels_from_list(bgr)

    # Transform
    def get_resize_image(self, width:int, height:int):
        if self.is_enable():
            return base.resize(self.image, (width, height))
        return None
    def get_rotate_image(self, angle:float):
        if self.is_invalid():
            return None
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = base.getRotationMatrix2D(center, angle, 1.0)
        return base.warpAffine(self.image, M, (w, h))
    def resize_image(self, width:int, height:int):
        """调整图片大小"""
        new_image = self.get_resize_image(width, height)
        if new_image is not None:
            self.image = new_image
        return self
    def rotate_image(self, angle:float):
        """旋转图片"""
        new_image = self.get_rotate_image(angle)
        if new_image is not None:
            self.image = new_image
        return self
    
    # 图片翻折
    def flip(self, flip_code:int):
        """翻转图片"""
        if self.is_enable():
            self.image = base.flip(self.image, flip_code)
        return self
    def horizon_flip(self):
        """水平翻转图片"""
        return self.flip(1)
    def vertical_flip(self):
        """垂直翻转图片"""
        return self.flip(0)
    def both_flip(self):
        """双向翻转图片"""
        return self.flip(-1)

    # 色彩空间猜测
    def guess_color_space(self) -> Optional[str]:
        """猜测色彩空间"""
        if self.is_invalid():
            return None
        image = self.image
        # 计算每个通道的像素值分布
        hist_b = base.calcHist([image], [0], None, [256], [0, 256])
        hist_g = base.calcHist([image], [1], None, [256], [0, 256])
        hist_r = base.calcHist([image], [2], None, [256], [0, 256])

        # 计算每个通道的像素值总和
        sum_b = np.sum(hist_b)
        sum_g = np.sum(hist_g)
        sum_r = np.sum(hist_r)

        # 根据像素值总和判断色彩空间
        if sum_b > sum_g and sum_b > sum_r:
            #print("The image might be in BGR color space.")
            return "BGR"
        elif sum_g > sum_b and sum_g > sum_r:
            #print("The image might be in GRAY color space.")
            return "GRAY"
        else:
            #print("The image might be in RGB color space.")
            return "RGB"

    # 颜色转化
    def get_convert(self, color_convert:int):
        """颜色转化"""
        if self.is_invalid():
            return None
        return base.cvtColor(self.image, color_convert)
    def convert_to(self, color_convert:int):
        """颜色转化"""
        if self.is_invalid():
            return None
        self.image = self.get_convert(color_convert)
    
    def is_grayscale(self):
        return self.dimension == 2
    def get_grayscale(self):
        if self.is_invalid():
            return None
        return base.cvtColor(self.image, base.COLOR_BGR2GRAY)
    def convert_to_grayscale(self):
        """将图片转换为灰度图"""
        self.image = self.get_grayscale()
        return self

    def get_convert_flag(
        self, 
        targetColorTypeName:Literal[
            "BGR", "RGB", "GRAY", "YCrCb"
            ]
        ) -> Optional[int]:
        """获取颜色转化标志"""
        flag = self.guess_color_space()
        if flag is None:
            return None
        
        if targetColorTypeName == "BGR":
            if flag == "RGB":
                return base.COLOR_RGB2BGR
            elif flag == "GRAY":
                return base.COLOR_GRAY2BGR
            elif flag == "YCrCb":
                return base.COLOR_YCrCb2BGR
        elif targetColorTypeName == "RGB":
            if flag == "BGR":
                return base.COLOR_BGR2RGB
            elif flag == "GRAY":
                return base.COLOR_GRAY2RGB
            elif flag == "YCrCb":
                return base.COLOR_YCrCb2RGB
        elif targetColorTypeName == "GRAY":
            if flag == "RGB":
                return base.COLOR_RGB2GRAY
            elif flag == "RGB":
                return base.COLOR_BGR2GRAY
        return None

    # 原址裁切
    def sub_image(self, x:int, y:int ,width:int ,height:int):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[y:y+height, x:x+width]
        return self

    # 直方图
    def equalizeHist(self, is_cover = False) -> MatLike:
        """直方图均衡化"""
        if self.is_invalid():
            return self
        result:MatLike = base.equalizeHist(self.image)
        if is_cover:
            self.image = result
        return result
    def calcHist(
        self, 
        channel:    Union[List[int], int],
        mask:       Optional[MatLike]       = None,
        hist_size:  Sequence[int]           = [256],
        ranges:     Sequence[float]         = [0, 256]
        ) -> MatLike:
        """计算直方图"""
        if self.is_invalid():
            return None
        return base.calcHist(
            [self.image],
            channel if isinstance(channel, list) else [channel],
            mask, 
            hist_size, 
            ranges)

    # 子集操作
    def sub_image_with_rect(self, rect:Tuple[float, float, float, float]):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return self
    def sub_image_with_box(self, box:Tuple[float, float, float, float]):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[box[1]:box[3], box[0]:box[2]]
        return self
    def sub_cover_with_rect(self, image:Union[Self, MatLike], rect:Tuple[float, float, float, float]):
        """覆盖图片"""
        if self.is_invalid():
            raise ValueError("Real Image is none")
        if isinstance(image, MatLike):
            image = ImageObject(image)
        self.image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = image.image
        return self
    def sub_cover_with_box(self, image:Union[Self, MatLike], box:Tuple[float, float, float, float]):
        """覆盖图片"""
        if self.is_invalid():
            raise ValueError("Real Image is none")
        if isinstance(image, MatLike):
            image = ImageObject(image)
        self.image[box[1]:box[3], box[0]:box[2]] = image.image
        return self

    def operator_cv(self, func:Callable[[MatLike], Any], *args, **kwargs):
        func(self.image, *args, **kwargs)
        return self

    def stack(self, *args:Self, **kwargs) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.stack([np.uint8(image.image) for image in images], *args, **kwargs))
    def vstack(self, *args:Self) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.vstack([np.uint8(image.image) for image in images]))
    def hstack(self, *args:Self) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.hstack([np.uint8(image.image) for image in images]))
    
    def add(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.add(self.image, image_or_value)
        else:
            self.image = base.add(self.image, image_or_value.image)
        return self
    def __add__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).add(image_or_value)
    def subtract(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.subtract(self.image, image_or_value)
        else:
            self.image = base.subtract(self.image, image_or_value.image)
        return self
    def __sub__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).subtract(image_or_value)
    def multiply(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.multiply(self.image, image_or_value)
        else:
            self.image = base.multiply(self.image, image_or_value.image)
        return self
    def __mul__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).multiply(image_or_value)
    def divide(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.divide(self.image, image_or_value)
        else:
            self.image = base.divide(self.image, image_or_value.image)
        return self
    def __truediv__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).divide(image_or_value)
    def bitwise_and(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.bitwise_and(self.image, image_or_value)
        else:
            self.image = base.bitwise_and(self.image, image_or_value.image)
        return self
    def bitwise_or(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = base.bitwise_or(self.image, image_or_value)
        else:
            self.image = base.bitwise_or(self.image, image_or_value.image)
        return self
    def bitwise_xor(self, image_or_value:Union[Self]):
        if isinstance(image_or_value, int):
            self.image = base.bitwise_xor(self.image, image_or_value)
        else:
            self.image = base.bitwise_xor(self.image, image_or_value.image)
        return self
    def bitwise_not(self):
        self.image = base.bitwise_not(self.image)
        return self
    def __neg__(self):
        return ImageObject(self.image.copy()).bitwise_not()
    
class NoiseImageObject(ImageObject):
    def __init__(
        self,
        height:     int,
        weight:     int,
        *,
        mean:       float   = 0,
        sigma:      float   = 25,
        dtype               = np.uint8
        ):
        noise = np.zeros((height,weight),dtype=dtype)
        base.randn(noise, mean, sigma)
        noise_bgr = base.cvtColor(noise, base.COLOR_GRAY2BGR)
        super().__init__(noise_bgr)

def Unwrapper(image:Optional[Union[
            str,
            ImageObject,
            tool_file, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]) -> MatLike:
    return image.image if isinstance(image, ImageObject) else ImageObject(image).image

def Wrapper(image:Optional[Union[
            str,
            ImageObject,
            tool_file, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]) -> ImageObject:
    return ImageObject(image)

class light_cv_window:
    def __init__(self, name:str):
        self.__my_window_name = name
        base.namedWindow(self.__my_window_name)
    def __del__(self):
        self.destroy()

    def show_image(self, image:Union[ImageObject, MatLike]):
        if self.__my_window_name is None:
            self.__my_window_name = "window"
        if isinstance(image, ImageObject):
            image = image.image
        base.imshow(self.__my_window_name, image)
        return self
    def destroy(self):
        if self.__my_window_name is not None and base.getWindowProperty(self.__my_window_name, base.WND_PROP_VISIBLE) > 0:
            base.destroyWindow(self.__my_window_name)
        return self
    
    @property
    def window_rect(self):
        return base.getWindowImageRect(self.__my_window_name)
    @window_rect.setter
    def window_rect(self, rect:Tuple[float, float, float, float]):
        self.set_window_rect(rect[0], rect[1], rect[2], rect[3])
    
    def set_window_size(self, weight:int, height:int):
        base.resizeWindow(self.__my_window_name, weight, height)
        return self
    def get_window_size(self) -> Tuple[float, float]:
        rect = self.window_rect
        return rect[2], rect[3]
    
    def get_window_property(self, prop_id:int):
        return base.getWindowProperty(self.__my_window_name, prop_id)
    def set_window_property(self, prop_id:int, prop_value:int):
        base.setWindowProperty(self.__my_window_name, prop_id, prop_value)
        return self
    def get_prop_frame_width(self):
        return self.window_rect[2]
    def get_prop_frame_height(self):
        return self.window_rect[3]
    def is_full_window(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN) > 0
    def set_full_window(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN, 1)
        return self
    def set_normal_window(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_FULLSCREEN, 0)
        return self
    def is_using_openGL(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_OPENGL) > 0
    def set_using_openGL(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_OPENGL, 1)
        return self
    def set_not_using_openGL(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_OPENGL, 0)
        return self
    def is_autosize(self):
        return base.getWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE) > 0
    def set_autosize(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE, 1)
        return self
    def set_not_autosize(self):
        base.setWindowProperty(self.__my_window_name, base.WINDOW_AUTOSIZE, 0)
        return self
    
    def set_window_rect(self, x:int, y:int, weight:int, height:int):
        base.moveWindow(self.__my_window_name, x, y)
        return self.set_window_size(weight, height)

    def set_window_pos(self, x:int, y:int):
        base.moveWindow(self.__my_window_name, x, y)
        return self

    def wait_key(self, wait_time:int=0):
        return base.waitKey(wait_time)

def get_haarcascade_frontalface(name_or_default:Optional[str]=None):
    if name_or_default is None:
        name_or_default = "haarcascade_frontalface_default"
    return base.CascadeClassifier(BaseData.haarcascades+'haarcascade_frontalface_default.xml')

def detect_human_face(
    image:          ImageObject,
    detecter:       base.CascadeClassifier, 
    scaleFactor:    float                   = 1.1,
    minNeighbors:   int                     = 4,
    *args, **kwargs):
    '''return is Rect[]'''
    return detecter.detectMultiScale(image.image, scaleFactor, minNeighbors, *args, **kwargs)

class internal_detect_faces_oop(Callable[[ImageObject], None]):
    def __init__(self):
        self.face_cascade = get_haarcascade_frontalface()
    def __call__(self, image:ImageObject):
        gray = image.convert_to_grayscale()
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            image.operator_cv(base.rectangle,(x,y),(x+w,y+h),(255,0,0),2)
    
def easy_detect_faces(camera:light_cv_camera):
    ImageObject(camera).show_image("window", 'q', internal_detect_faces_oop())
    
# 示例使用
if __name__ == "__main__":
    img_obj = ImageObject("path/to/your/image.jpg")
    img_obj.show_image()
    img_obj.resize_image(800, 600)
    img_obj.rotate_image(45)
    img_obj.convert_to_grayscale()
    img_obj.save_image("path/to/save/image.jpg")





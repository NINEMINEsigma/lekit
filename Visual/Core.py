from typing                 import *
from pydantic               import BaseModel
from abc                    import *
import                             random
import numpy                as     np
import matplotlib.pyplot    as     plt
import seaborn              as     sns
import                             torch
from lekit.Internal         import *
from lekit.Str.Core         import UnWrapper as Unwrapper2Str
from lekit.File.Core        import tool_file, Wrapper as Wrapper2File, tool_file_or_str, is_image_file
from lekit.Visual.OpenCV    import ImageObject, tool_file_cvex, WrapperFile2CVEX, Wrapper as Wrapper2Image
from PIL.Image              import Image as PILImage
from PIL.ImageFile          import ImageFile as PILImageFile

class data_visual_generator:
    def __init__(self, file:tool_file_or_str):
        self._file:tool_file = Wrapper2File(file)
        self._file.load()

    def open(self, mode='r', is_refresh=False, encoding:str='utf-8', *args, **kwargs):
        self._file.open(mode, is_refresh, encoding, *args, **kwargs)

    def reload(self, file:Optional[tool_file_or_str]):
        if file is not None:
            self._file = Wrapper2File(file)
        self._file.load()


    def plot_line(self, x, y, df=None, title="折线图", x_label=None, y_label=None):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df if df is not None else self._file.data, x=x, y=y)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.grid(True)
        plt.show()

    def plot_bar(self, x, y, df=None, figsize=(10,6), title="柱状图", x_label=None, y_label=None):
        plt.figure(figsize=figsize)
        sns.barplot(data=df if df is not None else self._file.data, x=x, y=y)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.grid(True)
        plt.show()

    def plot_scatter(self, x, y, df=None, title="散点图", x_label=None, y_label=None):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df if df is not None else self._file.data, x=x, y=y)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.grid(True)
        plt.show()

    def plot_histogram(self, column, df=None, title="直方图", x_label=None, y_label=None):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df if df is not None else self._file.data, x=column)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(column))
        plt.ylabel(y_label if y_label is not None else "value")
        plt.grid(True)
        plt.show()

    def plot_pairplot(self, df=None, title="成对关系图"):
        sns.pairplot(df if df is not None else self._file.data)
        plt.suptitle(title, y=1.02)
        plt.show()

    def plot_pie(self, column, figsize=(10,6), df=None, title="饼图"):
        plt.figure(figsize=figsize)
        if df is not None:
            df[column].value_counts().plot.pie(autopct='%1.1f%%')
        else:
            self._file.data[column].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(title)
        plt.ylabel('')  # 移除y轴标签
        plt.show()

    def plot_box(self, x, y, df=None, figsize=(10,6), title="箱线图", x_label=None, y_label=None):
        plt.figure(figsize=figsize)
        sns.boxplot(data=df if df is not None else self._file.data, x=x, y=y)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.grid(True)
        plt.show()

    def plot_heatmap(self, df=None, figsize=(10,6), title="热力图", cmap='coolwarm'):
        plt.figure(figsize=figsize)
        sns.heatmap(df.corr() if df is not None else self._file.data.corr(), annot=True, cmap=cmap)
        plt.title(title)
        plt.show()

    def plot_catplot(self, x, y, hue=None, df=None, kind='bar', figsize=(10,6), title="分类数据图", x_label=None, y_label=None):
        plt.figure(figsize=figsize)
        sns.catplot(data=df if df is not None else self._file.data, x=x, y=y, hue=hue, kind=kind)
        plt.title(title)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.grid(True)
        plt.show()
    def plot_catplot_strip(self, x, y, hue=None, df=None, figsize=(10,6), title="分类数据图", x_label=None, y_label=None):
        self.plot_catplot(x, y, hue=hue, df=df, kind='strip', figsize=figsize, title=title, x_label=x_label, y_label=y_label)
    def plot_catplot_swarm(self, x, y, hue=None, df=None, figsize=(10,6), title="分类数据图", x_label=None, y_label=None):
        self.plot_catplot(x, y, hue=hue, df=df, kind='swarm', figsize=figsize, title=title, x_label=x_label, y_label=y_label)
    def plot_catplot_box(self, x, y, hue=None, df=None, figsize=(10,6), title="分类数据图", x_label=None, y_label=None):
        self.plot_catplot(x, y, hue=hue, df=df, kind='box', figsize=figsize, title=title, x_label=x_label, y_label=y_label)
    def plot_catplot_violin(self, x, y, hue=None, df=None, figsize=(10,6), title="分类数据图", x_label=None, y_label=None):
        self.plot_catplot(x, y, hue=hue, df=df, kind='violin', figsize=figsize, title=title, x_label=x_label, y_label=y_label)

    def plot_jointplot(self, x, y, kind="scatter", df=None, title="联合图", x_label=None, y_label=None):
        sns.jointplot(data=df if df is not None else self._file.data, x=x, y=y, kind=kind)
        plt.suptitle(title, y=1.02)
        plt.xlabel(x_label if x_label is not None else str(x))
        plt.ylabel(y_label if y_label is not None else str(y))
        plt.show()
    def plot_jointplot_scatter(self, x, y, df=None, title="联合图", x_label=None, y_label=None):
        self.plot_jointplot(x, y, kind="scatter", df=df, title=title, x_label=x_label, y_label=y_label)
    def plot_jointplot_kde(self, x, y, df=None, title="联合图", x_label=None, y_label=None):
        self.plot_jointplot(x, y, kind="kde", df=df, title=title, x_label=x_label, y_label=y_label)
    def plot_jointplot_hex(self, x, y, df=None, title="联合图", x_label=None, y_label=None):
        self.plot_jointplot(x, y, kind="hex", df=df, title=title, x_label=x_label, y_label=y_label)

class data_math_virsual_generator(data_visual_generator):
    def drop_missing_values(self, axis):
        """删除缺失值"""
        self._file.data = self._file.data.dropna(axis=axis)

    def fill_missing_values(self, value):
        """填充缺失值"""
        self._file.data = self._file.data.fillna(value)

    def remove_duplicates(self):
        """删除重复值"""
        self._file.data = self._file.data.drop_duplicates()

    def standardize_data(self):
        """数据标准化"""
        self._file.data = (self._file.data - self._file.data.mean()) / self._file.data.std()

    def normalize_data(self):
        """数据归一化"""
        self._file.data = (self._file.data - self._file.data.min()) / (self._file.data.max() - self._file.data.min())

# region image augmentation

class BasicAugmentConfig(BaseModel, ABC):
    name:       Optional[str]                   = None
    @abstractmethod
    def augment(
        self,
        origin:     ImageObject
        ) -> Tuple[Dict[str, Any], ImageObject]:
        '''
        result:
            (change config, image)
        '''
        raise NotImplementedError()
class ResizeAugmentConfig(BasicAugmentConfig):
    width:      Optional[int]                   = None
    height:     Optional[int]                   = None
    name:       Optional[str]                   = "resize"
    @override
    def augment(
        self,
        origin:     ImageObject
        ) -> Tuple[Dict[str, Any], ImageObject]:
        width = self.width
        height = self.height
        if width is None and height is None:
            rangewidth = origin.width
            rangeheight = origin.height
            width = rangewidth + random.randint(
                (-rangewidth*(random.random()%1)).__floor__(),
                (rangewidth*(random.random()%1)).__floor__()
                )
            height = rangeheight + random.randint(
                (-rangeheight*(random.random()%1)).__floor__(),
                (rangeheight*(random.random()%1)).__floor__()
                )
        elif width is None:
            width = origin.width
        elif height is None:
            height = origin.height
        change_config = {
            "width":width,
            "height":height
        }
        return (change_config, ImageObject(origin.get_resize_image(abs(width), abs(height))))
# Config.name -> (field, value)
type ChangeConfig = Dict[str, Dict[str, Any]]
# (field, value)
type ResultImageObjects = Dict[str, ImageObject]
class ImageAugmentConfig(BaseModel):
    resize:     Optional[ResizeAugmentConfig]   = None
    log_call:   Optional[Callable[[Union[str, Dict[str, Any]]], None]] = None

    def _inject_log(self, *args, **kwargs):
        if self.log_call is not None:
            self.log_call(*args, **kwargs)

    def augment(
        self,
        origin:     ImageObject
        ) -> Tuple[ChangeConfig, ResultImageObjects]:
        result:                 Dict[str, ImageObject]      = {}
        result_change_config:   Dict[str, Dict[str, Any]]   = {}
        augment_configs:        List[BasicAugmentConfig]    = [
            self.resize
        ]
        for item in augment_configs:
            if item is not None:
                result_change_config[item.name], result[item.name] = self.resize.augment(origin)
                self._inject_log(f"augmentation<{item.name}> change config: {result_change_config[item.name]}")
        return (result_change_config, result)
    def augment_to(
            self,
            input:      Union[tool_file, str, ImageObject, np.ndarray, PILImage, PILImageFile],
            output_dir: tool_file_or_str,
            *,
            # if output_dir is not exist, it will call must_exist
            # if output_dir is exist but not dir, it will back to parent
            must_output_dir_exist:  bool                                = False,
            output_file_name:       str                                 = "output.png",
            callback:               Optional[Action[ChangeConfig]]      = None,
        ) -> ResultImageObjects:
        # Init env and vars
        origin_image:   ImageObject = self.__init_origin_image(input)
        result_dir:     tool_file   = self.__init_result_dir(output_dir, must_output_dir_exist)
        # augment
        self._inject_log(f"output<{output_file_name}> is start augment")
        change_config, result = self._inject_augment(
            origin_image=origin_image,
            result_dir=result_dir,
            output_file_name=output_file_name,
        )
        # result
        if callback is not None:
            callback(change_config)
        return result
    def augment_from_dir_to(
            self,
            input_dir:  Union[tool_file, str],
            output_dir: tool_file_or_str,
            *,
            # if output_dir is not exist, it will call must_exist
            # if output_dir is exist but not dir, it will back to parent
            must_output_dir_exist:  bool                                        = False,
            callback:               Optional[Action2[tool_file, ChangeConfig]]  = None,
        ) -> Dict[str, List[ImageObject]]:
        # Init env and vars
        origin_images:  tool_file   = Wrapper2File(input_dir)
        result_dir:     tool_file   = self.__init_result_dir(output_dir, must_output_dir_exist)
        if origin_images.exists() is False or origin_images.is_dir() is False:
            raise FileExistsError(f"input_dir<{origin_images}> is not exist or not dir")
        # augment
        result: Dict[str, List[ImageObject]] = {}
        for image_file in origin_images.dir_tool_file_iter():
            if is_image_file(Unwrapper2Str(image_file)) is False:
                continue
            change_config, curResult = self._inject_augment(
                origin_image=WrapperFile2CVEX(image_file).load(),
                result_dir=result_dir,
                output_file_name=image_file.get_filename(),
            )
            # append single result
            for key in curResult:
                if key in result:
                    result[key].append(curResult[key])
                else:
                    result[key] = [curResult[key]]
            # call feedback
            if callback is not None:
                callback(image_file, change_config)
        # result
        return result
    def augment_from_images_to(
            self,
            inputs:     Sequence[ImageObject],
            output_dir: tool_file_or_str,
            *,
            # if output_dir is not exist, it will call must_exist
            # if output_dir is exist but not dir, it will back to parent
            must_output_dir_exist:  bool                                            = False,
            callback:               Optional[Action2[ImageObject, ChangeConfig]]    = None,
            fileformat:             str                                             = "{}.jpg",
            indexbuilder:           type                                            = int
        ) -> Dict[str, List[ImageObject]]:
        # Init env and vars
        result_dir:     tool_file   = self.__init_result_dir(output_dir, must_output_dir_exist)
        index:          Any         = indexbuilder()
        # augment
        result: Dict[str, List[ImageObject]] = {}
        for image in inputs:
            current_output_name = fileformat.format(index)
            change_config, curResult = self._inject_augment(
                origin_image=image,
                result_dir=result_dir,
                output_file_name=current_output_name,
            )
            # append single result
            for key in curResult:
                result[key].append(curResult[key])
            index += 1
            # call feedback
            if callback is not None:
                callback(image, change_config)
        # result
        return result
    def __init_origin_image(self, input:Union[tool_file, str, ImageObject, np.ndarray, PILImage, PILImageFile]) -> ImageObject:
        origin_image:   ImageObject = None
        # check
        if isinstance(input, (tool_file, str)):
            inputfile = WrapperFile2CVEX(input)
            if inputfile.data is not None:
                origin_image = inputfile.data
            else:
                origin_image = inputfile.load()
        elif isinstance(input, (ImageObject, np.ndarray, PILImage, PILImageFile)):
            origin_image = Wrapper2Image(input)
        else:
            raise TypeError(f"input<{input}> is not support type")
        return origin_image
    def __init_result_dir(self, output_dir:tool_file_or_str, must_output_dir_exist:bool) -> tool_file:
        result_dir:     tool_file   = Wrapper2File(output_dir)
        # check exist
        stats:          bool        = True
        if result_dir.exists() is False:
            if must_output_dir_exist:
                result_dir.must_exists_path()
            else:
                stats = False
        if stats is False:
            raise FileExistsError(f"output_dir<{result_dir}> is not exist")
        # check dir stats
        if result_dir.is_dir() is False:
            if must_output_dir_exist:
                result_dir.back_to_parent_dir()
            else:
                raise FileExistsError(f"output_dir<{result_dir}> is not a dir")
        # result
        return result_dir

    def _inject_augment(
        self,
        origin_image:               ImageObject,
        result_dir:                 tool_file,
        output_file_name:           str
        ) -> Tuple[ChangeConfig, ResultImageObjects]:
        result_dict, result_images = self.augment(origin_image)
        self._inject_log(f"output<{output_file_name}> is start augment")
        for key, value in result_images.items():
            current_dir = result_dir|key
            current_result_file = current_dir|output_file_name
            value.save_image(current_result_file, True)
        return result_dict, result_images

# region end



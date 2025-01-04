from typing                 import *
import matplotlib.pyplot    as     plt
import seaborn              as     sns
from lekit.File.Core        import tool_file, Wrapper as Wrapper2File, tool_file_or_str

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



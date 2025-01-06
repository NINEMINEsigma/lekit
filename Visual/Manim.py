# -*- coding: utf-8 -*-
from lekit.Internal         import *
from lekit.MathEx.Core      import *
from lekit.File.Core        import tool_file
import manim                as     base
from manim.mobject.mobject  import Mobject
from manim                  import typing as MTypes

# init env
def init_manim_dir_env():
    with tool_file("media/").must_exists_path() as assets:
        (assets|"videos/").must_exists_path()
        (assets|"images/").must_exists_path()
        (assets|"Tex/").must_exists_path()
if False:
    init_manim_dir_env()

VectorFrontward = base.OUT
VectorBackward = base.IN
Point3DTuple = Tuple[float, float, float]
def Wrapper2MPoint3D(
    datahead_or_x:  Union[NumberLike, Sequence[NumberLike], np.ndarray],
    y:              NumberLike = 0,
    z:              NumberLike = 0,
    ) -> MTypes.Point3D:
    point:np.ndarray = None
    if isinstance(datahead_or_x, NumberLike):
        point = np.array((datahead_or_x, y, z))
    elif isinstance(datahead_or_x, np.ndarray):
        return datahead_or_x
    elif isinstance(datahead_or_x, Sequence):
        length = len(datahead_or_x)
        if length == 2:
            point = np.array((datahead_or_x[0], datahead_or_x[1], 0))
        elif length >= 3:
            point = np.array((datahead_or_x[0], datahead_or_x[1], datahead_or_x[2]))
        else:
            raise ValueError(f"When datahead_or_x is sequence, it must be a 2D-point or 3D-point, but current length is {length}")
    else:
        raise ValueError("datahead_or_x must be a number, a 3D-point or a sequence of numbers")
    return point.astype(np.float64)
def Wrapper2MPoint2D(
    datahead_or_x:  Union[NumberLike, Sequence[NumberLike], np.ndarray],
    y:              NumberLike = 0,
    ) -> MTypes.Point2D:
    point:np.ndarray = None
    if isinstance(datahead_or_x, NumberLike):
        point = np.array((datahead_or_x, y))
    elif isinstance(datahead_or_x, np.ndarray):
        return datahead_or_x
    elif isinstance(datahead_or_x, Sequence):
        length = len(datahead_or_x)
        if length == 2:
            point = np.array((datahead_or_x[0], datahead_or_x[1], 0))
        else:
            raise ValueError(f"When datahead_or_x is sequence, it must be a 2D-point, but current length is {length}")
    else:
        raise ValueError("datahead_or_x must be a number, a 2D-point or a sequence of numbers")
    return point.astype(np.float64)
enable_unwrapper2center_type = Union[Point3DTuple, Mobject, left_value_reference[Mobject]]
def Unwrapper2Center(center:enable_unwrapper2center_type= (0, 0, 0)) -> Union[Point3DTuple, Mobject]:
    center_object_or_point = None
    if isinstance(center, Mobject):
        center_object_or_point = center
    elif isinstance(center, left_value_reference):
        center_object_or_point = center.ref_value
    else:
        center_object_or_point = Wrapper2MPoint3D(center)
    return center_object_or_point

class _TransformExample(base.Scene):
    def construct(self):

        banner = base.ManimBanner()
        banner.shift(base.UP * 0.5)
        self.play(banner.create(), run_time=1)
        self.play(banner.animate.scale(0.3), run_time=0.5)
        self.play(banner.expand(), run_time=1)

        t = base.Text("测试中文能否显示").next_to(banner, base.DOWN * 2)
        tex = base.VGroup(
            base.Text("测试数学公式:", font_size=30),
            base.Tex(r"$\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$"),
        )
        tex.arrange(base.RIGHT, buff=base.SMALL_BUFF)
        tex.next_to(t, base.DOWN)
        self.play(base.Write(t), run_time=1)
        self.play(base.Write(tex), run_time=1)

        self.wait()

CDstructorType = Action2[Self, base.Scene]

class AnimationConfig:
    # delay_and_wait_until_pr -> delay(max time), wait_until_pr
    delay_and_wait_until_pr:    Tuple[float, Optional[Callable[[], bool]]]                  = (0, None)
    @property
    def delay_or_max_time(self) -> float:
        return self.delay_and_wait_until_pr[0]
    @property
    def wait_until(self) -> Optional[Callable[[], bool]]:
        return self.delay_and_wait_until_pr[1]
    # scene.play(<this.property>(self.ref_value), **self.play_config) -- see in Animation.inject_play_animation
    animate_creater:            Union[base.Animation, Callable[[Mobject], base.Animation]]  = base.Create

    def __init__(
        self,
        delay_and_wait_until_pr:    Tuple[float, Optional[Callable[[], bool]]]                  = (0, None),
        animate_creater:            Union[base.Animation, Callable[[Mobject], base.Animation]]  = base.Create,
        ):
        self.delay_and_wait_until_pr = delay_and_wait_until_pr
        self.animate_creater = animate_creater
    def copy(self):
        return AnimationConfig(self.delay_and_wait_until_pr, self.animate_creater)
class Animation(left_value_reference[Mobject]):
    def copy(self):
        return type(self)(self.ref_value,
            self.play_config["run_time"],
            argsconfig=self.args_config.copy(),
            playconfig=self.play_config.copy(),
            constructor=self.cd_constructor,
            destructor=self.cd_destructor
            )
    def __init__(
        self,
        element:                    Mobject,
        duration:                   NumberLike,
        *,
        argsconfig:                 Optional[AnimationConfig]   = None,
        playconfig:                 Optional[Dict[str, Any]]    = None,
        constructor:                CDstructorType              = None,
        destructor:                 CDstructorType              = None,
        ):
        '''
        element:
            target Mobject
        duration:
            run_time
        '''
        # Build up env
        super().__init__(element)
        self.args_config:       AnimationConfig     = argsconfig if argsconfig is not None else AnimationConfig()
        self.play_config:       Dict[str, Any]      = playconfig if playconfig is not None else {}
        self.play_config["run_time"] = duration
        # Build up self var
        self.m_animation_instance:  base.Animation              = None
        self.cd_constructor:        CDstructorType              = constructor
        self.cd_destructor:         CDstructorType              = destructor
        self.__cd_stats:            bool                        = False
    def inject_play_animation(self, scene:base.Scene):
        if self.m_animation_instance is None:
            self.m_animation_instance = self.args_config.animate_creater(self.ref_value)
        scene.play(self.m_animation_instance, **self.play_config)
    def __catch_cd(self, scene):
        if self.__cd_stats is False:
            if self.cd_constructor is not None:
                self.cd_constructor(self, scene)
            self.__cd_stats = True
    def __release_cd(self, scene):
        if self.__cd_stats is True:
            if self.cd_destructor is not None:
                self.cd_destructor(self, scene)
            self.__cd_stats = False
    def play_animation(self, scene:base.Scene):
        self.__catch_cd(scene)
        self.activate_scene = scene
        self.inject_play_animation(scene)
        # Check delay stats
        if self.args_config.wait_until is not None:
            scene.wait_until(self.args_config.wait_until, self.args_config.delay_or_max_time)
        elif self.args_config.delay_or_max_time > 0:
            scene.wait(self.args_config.delay_or_max_time)
    def release_play_animation(self, scene:base.Scene):
        self.__release_cd(scene)
    @property
    def cd_stats(self):
        return self.__cd_stats

    def move_to(
        self,
        to_point:   Point3DTuple
        ) -> Self:
        if self:
            self.ref_value.move_to(Wrapper2MPoint3D(to_point))
        return self
    def rotate(
        self,
        angle:          float,
        about_point:    Point3DTuple    = base.ORIGIN,
        axis:           MTypes.Vector3D = VectorFrontward
        ) -> Self:
        if self:
            self.ref_value.rotate(angle=angle, axis=axis, about_point=Wrapper2MPoint3D(about_point))
        return self
    def scale(
        self,
        scale_factor:   float,
        about_point:    Point3DTuple    = base.ORIGIN
        ) -> Self:
        if self:
            self.ref_value.scale(scale_factor=scale_factor, about_point=Wrapper2MPoint3D(about_point))
        return self
class WaitAnimation(Animation):
    def __init__(
        self,
        duration:           float,
        stop_condition:     Optional[Callable[[], bool]]    = None,
        frozen_frame:       Optional[bool]                  = None,
        rate_func:          Callable[[float], float]        = base.linear,
        **kwargs
        ):
        super().__init__(None, duration)
        self.m_animation_instance = base.Wait(duration, stop_condition, frozen_frame, rate_func, **kwargs)
    @override
    def release_play_animation(self, scene):
        pass

def WrapperMobjects2Animations(
    objs:Union[Union[Animation, Mobject], Sequence[Union[Animation, Mobject]]],
    duration:float,
    *,
    argsconfig:                 Optional[AnimationConfig]   = None,
    playconfig:                 Optional[Dict[str, Any]]    = None,
    constructor:                CDstructorType              = None,
    destructor:                 CDstructorType              = None,
    maker:                      Optional[Union[type, Callable]] = None
    ) -> List[Animation]:
    if maker is None:
        maker = Animation
    if isinstance(objs, Sequence) is False:
        objs = [objs]
    result:List[Animation] = []
    for obj in objs:
        if isinstance(obj, Animation) is False:
            obj = maker(obj, duration,
                        argsconfig=argsconfig, playconfig=playconfig,
                        constructor=constructor, destructor=destructor)
        result.append(obj)
    return result

# Make point
def make_point(
    point:          MTypes.Point3D,
    radius:         float = base.DEFAULT_DOT_RADIUS,
    stroke_width:   float = 0,
    fill_opacity:   float = 1.0,
    color:          base.ParsableManimColor = base.WHITE,
    **kwargs
    ) -> base.Dot:
    '''
    Create animation of point:
        Animation(make_point(Wrapper2MPoint3D(point)), duration=0.5)
    '''
    return base.Dot(point, radius, stroke_width, fill_opacity, color, **kwargs)
# Make any object like line
def do_make_line(
    from_point:     Point3DTuple,
    to_point:       Point3DTuple,
    typen:          Union[type, Callable[[Point3DTuple, Point3DTuple], Union[Animation, Mobject]]],
    /,
    **kwargs,
    ) -> Union[Animation, Mobject]:
    return typen(Wrapper2MPoint3D(from_point), Wrapper2MPoint3D(to_point), **kwargs)
def make_line(
    from_point:     Point3DTuple,
    to_point:       Point3DTuple,
    buff:           float           = 0,
    path_arc:       Optional[float] = None,
    **kwargs,
    ) -> base.Line:
    '''
    Create animation of line:
        Animation(make_line(from_point, to_point), duration=0.5)
    '''
    return do_make_line(from_point, to_point, base.Line,
                        buff=buff, path_arc=path_arc, **kwargs)
def make_arrow(
    from_point:     Point3DTuple,
    to_point:       Point3DTuple,
    stroke_width:   float = 6,
    buff:           float = base.MED_SMALL_BUFF,
    max_tip_length_to_length_ratio:     float = 0.25,
    max_stroke_width_to_length_ratio:   float = 5,
    **kwargs,
    ) -> base.Arrow:
    '''
    Create animation of arrow:
        Animation(make_arrow(from_point, to_point), duration=0.5)
    '''
    return do_make_line(from_point, to_point, base.Arrow,
                        buff=buff, stroke_width=stroke_width, max_stroke_width_to_length_ratio=max_stroke_width_to_length_ratio,
                        max_tip_length_to_length_ratio=max_tip_length_to_length_ratio, **kwargs)
def make_dashedline(
    from_point:     Point3DTuple,
    to_point:       Point3DTuple,
    dash_length:    float = base.DEFAULT_DASH_LENGTH,
    dashed_ratio:   float = 0.5,
    **kwargs,
    ) -> base.DashedLine:
    '''
    Create animation of dashedline:
        Animation(make_dashedline(from_point, to_point), duration=0.5)
    '''
    return do_make_line(from_point, to_point, base.DashedLine,
                        dash_length=dash_length, dashed_ratio=dashed_ratio, **kwargs)
# Make circle
def make_circle(
    # transform
    center:         enable_unwrapper2center_type = (0, 0, 0),
    # style
    radius:         float                   = 1,
    **kwargs
    ) -> base.Circle:
    '''
    Create animation of circle:
        Animation(make_circle(center, radius), duration=0.5)
    '''
    return base.Circle(radius, **kwargs).move_to(Unwrapper2Center(center))
def make_ellipse(
    # transform
    center:         enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:          float                    = 0,
    rotate_axis:           MTypes.Vector3D          = base.OUT,
    rotate_about_point:    Optional[MTypes.Point3D] = None,
    # style
    width:          float                           = 1,
    height:         float                           = 1,
    **kwargs
    ) -> base.Ellipse:
    '''
    Create animation of ellipse:
        Animation(make_ellipse(...), duration=0.5)
    '''
    return base.Ellipse(width, height, **kwargs).move_to(Unwrapper2Center(center)).rotate(rotate_angle, rotate_axis, rotate_about_point)
def make_arc(
    # transform
    center:                 MTypes.Point3D = base.ORIGIN,
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    radius:         float = 1.0,
    start_angle:    float = 0,
    angle:          float = base.TAU / 4,
    num_components: int = 9,
    **kwargs
    ) -> base.Arc:
    '''
    Create animation of arc:
    '''
    return base.Arc(radius, start_angle, angle, num_components, Unwrapper2Center(center),**kwargs).rotate(rotate_angle, rotate_axis, rotate_about_point)
# Make polygon
def make_regular_polygram(
    num_vertices:   int,
    # transform
    center:         enable_unwrapper2center_type    = (0, 0, 0),
    start_angle:    Optional[float]                 = None,
    # style
    density:        int                             = 1,
    radius:         float                           = 1,
    **kwargs
    ) -> base.RegularPolygram:
    '''
    Create animation of triangle:
        Animation(make_triangle(...), duration=0.5)
    '''
    return base.RegularPolygram(num_vertices, density=density, radius=radius, start_angle=start_angle, **kwargs
        ).move_to(Unwrapper2Center(center))
def make_polygram(
    vertices:       Iterable[MTypes.Point3D]        = [],
    # transform
    center:         enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:          float                    = 0,
    rotate_axis:           MTypes.Vector3D          = base.OUT,
    rotate_about_point:    Optional[MTypes.Point3D] = None,
    # style
    color:          base.ParsableManimColor         = base.BLUE,
    **kwargs
    ) -> base.Polygram:
    '''
    Create animation of polygram:
        Animation(make_polygram(...), duration=0.5)
    '''
    vertex_groups = tuple(vertices)
    return base.Polygram(vertex_groups, color=color, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)
def make_square(
    side_length:            float,
    # transform
    center:             enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    **kwargs
    ) -> base.Square:
    '''
    Create animation of square:
        Animation(make_square(...), duration=0.5)
    '''
    return base.Square(side_length=side_length, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)
def make_rectangle(
    width:                  float,
    height:                 float,
    # transform
    center:             enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    **kwargs
    ) -> base.Rectangle:
    '''
    Create animation of rectangle:
        Animation(make_rectangle(...), duration=0.5)
    '''
    return base.Rectangle(width=width, height=height, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)
def make_rounded_rectangle(
    width:          float,
    height:         float,
    corner_radius:  float,
    # transform
    center:             enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    **kwargs
    ) -> base.RoundedRectangle:
    '''
    Create animation of rounded rectangle:
        Animation(make_rounded_rectangle(...), duration=0.5)
    '''
    return base.RoundedRectangle(width=width, height=height, corner_radius=corner_radius, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)
# Make Tex
def make_tex(
    text:           str,
    # transform
    center:             enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    **kwargs
    ) -> base.Tex:
    '''
    Create animation of tex:
        Animation(make_tex(...), duration=0.5)
    '''
    return base.Tex(text, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)
def make_text(
    text:           str,
    # transform
    center:             enable_unwrapper2center_type    = (0, 0, 0),
    rotate_angle:           float                       = 0,
    rotate_axis:            MTypes.Vector3D             = base.OUT,
    rotate_about_point:     Optional[MTypes.Point3D]    = None,
    # style
    **kwargs
    ) -> base.Text:
    '''
    Create animation of text:
        Animation(make_text(...), duration=0.5)
    '''
    return base.Text(text, **kwargs
                        ).move_to(Unwrapper2Center(center)
                        ).rotate(rotate_angle, rotate_axis, rotate_about_point)

Points_Generater_or_Iter = Union[
        # result: stats, current point
        Callable[[], Tuple[bool, Point3DTuple]],
        # args: iter of point3DTuple
        Iterable[Point3DTuple],
        ]
# Make lines-like animations
def do_make_lines_animations(
    points:         Points_Generater_or_Iter,
    maker:          Union[
        type,
        Callable[[Point3DTuple, Point3DTuple], Union[
            Animation,
            Mobject,
            Sequence[Union[Animation, Mobject]]
            ]]],
    is_closed:      bool            = False,
    duration:       Optional[float] = None,
    /,
    **kwargs,
    ) -> List[Animation]:
    '''
    Args:
        points:     点集合或是创建点集合的迭代器
        maker:      创建实例/动画/复数实例或动画的可调用对象
        is_closed:  是否闭合
        duration:   动画总时长
    '''
    # Init vars
    build_points:   List[Tuple[NumberLike, NumberLike, NumberLike]] = []
    if isinstance(points, Callable):
        stats, build_point = points()
        while stats:
            build_points.append(build_point)
            stats, build_point = points()
    else:
        build_points.extend(points)
    # Check args and build point
    if is_closed and build_points[-1] != build_points[0]:
        build_points.append(build_points[0])
    if duration is None:
        duration = len(build_points)-1
    per_duration = duration/(len(build_points)-1)
    # Build up target animations
    origin_results = [
        maker(build_points[i], build_points[i+1], **kwargs) for i in range(len(build_points)-1)
        ]
    result = []
    for line in origin_results:
        if isinstance(line, Animation):
            result.append(line)
        elif isinstance(line, Sequence):
            for curitem in line:
                if isinstance(curitem, Animation):
                    result.append(curitem)
                else:
                    result.append(Animation(curitem, per_duration))
        else:
            result.append(Animation(line, per_duration))
    return result
def make_lines_animations(
    points:         Points_Generater_or_Iter,
    is_closed:      bool            = False,
    duration:       Optional[float] = None,
    buff:           float           = 0,
    path_arc:       Optional[float] = None,
    **kwargs,
    ) -> List[Animation]:
    '''
    Create animations of lines and add it to Timeline:
        Timeline.add_animations(make_lines_animations(points))
    '''
    return do_make_lines_animations(points, make_line, is_closed, duration,
                                    buff=buff, path_arc=path_arc, **kwargs)

_ScaleBase = TypeVar("MType._ScaleBase")
# Make number line
def make_numberline(
    x_range:                        Optional[Sequence[float]]   = None,  # must be first
    length:                         Optional[float]             = None,
    unit_size:                      float                       = 1,
    # ticks
    include_ticks:                  bool                        = True,
    tick_size:                      float                       = 0.1,
    numbers_with_elongated_ticks:   Optional[Iterable[float]]   = None,
    longer_tick_multiple:           int                         = 2,
    exclude_origin_tick:            bool                        = False,
    # visuals
    rotation:                       float                       = 0,
    stroke_width:                   float                       = 2.0,
    # tip
    include_tip:                    bool                        = False,
    tip_width:                      float                       = base.DEFAULT_ARROW_TIP_LENGTH,
    tip_height:                     float                       = base.DEFAULT_ARROW_TIP_LENGTH,
    tip_shape:                    Optional[type[base.ArrowTip]] = None,
    # numbers/labels
    include_numbers:                bool                        = False,
    font_size:                      float                       = 36,
    label_direction:                Sequence[float]             = base.DOWN,
    label_constructor:              base.VMobject               = base.MathTex,
    scaling:                        _ScaleBase                  = base.LinearBase(),
    line_to_number_buff:            float                       = base.MED_SMALL_BUFF,
    decimal_number_config:          Optional[dict]              = None,
    numbers_to_exclude:             Optional[Iterable[float]]   = None,
    numbers_to_include:             Optional[Iterable[float]]   = None,
    **kwargs,
    ) -> base.NumberLine:
    return base.NumberLine(x_range=x_range,
        length=length,
        unit_size=unit_size,
        include_ticks=include_ticks,
        tick_size=tick_size,
        numbers_with_elongated_ticks=numbers_with_elongated_ticks,
        longer_tick_multiple=longer_tick_multiple,
        exclude_origin_tick=exclude_origin_tick,
        rotation=rotation,
        stroke_width=stroke_width,
        include_tip=include_tip,
        tip_width=tip_width,
        tip_height=tip_height,
        tip_shape=tip_shape,
        include_numbers=include_numbers,
        font_size=font_size,
        label_direction=label_direction,
        label_constructor=label_constructor,
        scaling=scaling,
        line_to_number_buff=line_to_number_buff,
        decimal_number_config=decimal_number_config,
        numbers_to_exclude=numbers_to_exclude,
        numbers_to_include=numbers_to_include,
        **kwargs)
def easy_numberline(
    start:      float,
    end:        float,
    length:     float,
    unit_size:  float,
    # transform
    rotation:   float = 0,
    # style
    font_size:  int = 24,
    **kwargs
    ) -> base.NumberLine:
    return base.NumberLine(x_range=[start, end],
        length=length,
        unit_size=unit_size,
        rotation=rotation,
        include_numbers=True,
        include_tip=True,
        font_size=font_size,
        **kwargs)

# Make CoordinateSystem
class AxesAnimation(Animation):
    def __init__(
        self,
        plane_element:  base.Axes,
        duration:       float,
        argsconfig:     Optional[AnimationConfig]           = None,
        playconfig:     Optional[Dict[str, Any]]            = None,
        constructor:    CDstructorType                      = None,
        destructor:     CDstructorType                      = None,
        ):
        super().__init__(plane_element, duration,
            argsconfig=argsconfig, playconfig=playconfig, constructor=constructor, destructor=destructor)
    @override
    def inject_play_animation(self, scene):
        return super().inject_play_animation(scene)
    @property
    def plane(self) -> base.Axes:
        return self.ref_value

    def local(
        self,
        *coords: Union[float, Sequence[float], Sequence[Sequence[float]], np.ndarray]
        ) -> np.ndarray:
        return self.plane.coords_to_point(*coords)
    def coords_to_point(self, *coords: float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self.plane.coords_to_point(*coords)
    def point_to_coords(self, point: Sequence[float]) -> np.ndarray:
        return self.plane.point_to_coords(point)

    def make_point(
        self,
        *coords:    Union[float, Sequence[float], Sequence[Sequence[float]], np.ndarray],
        point_or_point_config:      Optional[Union[base.Dot, Dict[str, Any]]] = None,
        tex_or_tex_config:          Optional[Union[base.Tex, Dict[str, Any]]] = None,
        label_toward:               MTypes.Vector3D                           = base.UR
        ) -> Tuple[base.Dot, base.Tex]:
        pos = self.local(*coords)
        point:  base.Dot = WrapperConfig2Instance(make_point, point_or_point_config,  datahead_typen_=base.Dot, point=pos)
        tex:    base.Tex = WrapperConfig2Instance(make_tex,   tex_or_tex_config,      datahead_typen_=base.Tex, text=f"({pos[0]},{pos[1]})")
        point.move_to(pos)
        tex.next_to(point, label_toward)
        return point, tex

class NumberPlaneAnimationConfig:
    def __init__(
        self,
        duration:           NumberLike,
        # basic
        x_range:            Optional[Sequence[NumberLike]]  = None,
        y_range:            Optional[Sequence[NumberLike]]  = None,
        x_length:           Optional[float]                 = None,
        y_length:           Optional[float]                 = None,
        numberplaneType:    base.NumberPlane = base.NumberPlane,
        # style
        background_line_style:  Optional[Dict[str, Any]]    = None,
        faded_line_style:       Optional[Dict[str, Any]]    = None,
        faded_line_ratio:       int                         = 1,
        make_smooth_after_applying_functions:          bool = True,
        **kwargs: dict[str, Any],
        ):
        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length
        self.numberplaneType = numberplaneType
        self.background_line_style = background_line_style
        self.faded_line_style = faded_line_style
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.kwargs = kwargs
        self.plane = numberplaneType(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            background_line_style=background_line_style,
            faded_line_style=faded_line_style,
            faded_line_ratio=faded_line_ratio,
            make_smooth_after_applying_functions=make_smooth_after_applying_functions,
            **kwargs)
        self.duration = duration
class NumberPlaneAnimation(AxesAnimation):
    def __init__(
        self,
        planeconfig:    Optional[NumberPlaneAnimationConfig] = None,
        argsconfig:     Optional[AnimationConfig]            = None,
        playconfig:     Optional[Dict[str, Any]]             = None,
        constructor:    CDstructorType                       = None,
        destructor:     CDstructorType                       = None,
        ):
        if planeconfig is None:
            planeconfig = NumberPlaneAnimationConfig()
        self.plane_config = planeconfig
        super().__init__(planeconfig.plane, planeconfig.duration,
                         argsconfig=argsconfig, playconfig=playconfig, constructor=constructor, destructor=destructor)

class PolarPlaneAnimationConfig:
    def __init__(
        self,
        duration,
        radius_max:                 Optional[float]             = None,
        size:                       Optional[float]             = None,
        radius_step:                float                       = 1,
        azimuth_step:               Optional[float]             = None,
        azimuth_units:              Optional[str]               = "PI radians",
        azimuth_compact_fraction:   bool                        = True,
        azimuth_offset:             float                       = 0,
        azimuth_direction:          str                         = "CCW",
        azimuth_label_buff:         float                       = base.SMALL_BUFF,
        azimuth_label_font_size:    float                       = 24,
        radius_config:              Optional[dict[str, Any]]    = None,
        background_line_style:      Optional[dict[str, Any]]    = None,
        faded_line_style:           Optional[dict[str, Any]]    = None,
        faded_line_ratio:           int                         = 1,
        make_smooth_after_applying_functions: bool              = True,
        **kwargs: Any,
        ):
        self.duration = duration
        self.radius_max = radius_max
        self.size = size
        self.radius_step = radius_step
        self.azimuth_step = azimuth_step
        self.azimuth_units = azimuth_units
        self.azimuth_compact_fraction = azimuth_compact_fraction
        self.azimuth_offset = azimuth_offset
        self.azimuth_direction = azimuth_direction
        self.azimuth_label_buff = azimuth_label_buff
        self.azimuth_label_font_size = azimuth_label_font_size
        self.radius_config = radius_config
        self.background_line_style = background_line_style
        self.faded_line_style = faded_line_style
        self.faded_line_ratio = faded_line_ratio
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.kwargs = kwargs
        if radius_max is not None:
            kwargs["radius_max"] = radius_max
        self.plane = base.PolarPlane(
            size=self.size,
            radius_step=self.radius_step,
            azimuth_step=self.azimuth_step,
            azimuth_units=self.azimuth_units,
            azimuth_compact_fraction=self.azimuth_compact_fraction,
            azimuth_offset=self.azimuth_offset,
            azimuth_direction=self.azimuth_direction,
            azimuth_label_buff=self.azimuth_label_buff,
            azimuth_label_font_size=self.azimuth_label_font_size,
            radius_config=self.radius_config,
            background_line_style=self.background_line_style,
            faded_line_style=self.faded_line_style,
            faded_line_ratio=self.faded_line_ratio,
            make_smooth_after_applying_functions=self.make_smooth_after_applying_functions,
            **kwargs
        )
class PolarPlaneAnimation(AxesAnimation):
    def __init__(
        self,
        planeconfig:    Optional[PolarPlaneAnimationConfig] = None,
        argsconfig:     Optional[AnimationConfig]           = None,
        playconfig:     Optional[Dict[str, Any]]            = None,
        constructor:    CDstructorType                      = None,
        destructor:     CDstructorType                      = None,
        ):
        if planeconfig is None:
            planeconfig = PolarPlaneAnimationConfig()
        self.plane_config = planeconfig
        super().__init__(planeconfig.plane, planeconfig.duration,
                         argsconfig=argsconfig, playconfig=playconfig, constructor=constructor, destructor=destructor)

class Timeline(base.ThreeDScene, Animation):
    def __init__(
        self,
        animations:             Optional[List[Animation]]   = None,
        *,
        # Scene Args
        renderer                                            = None,
        camera_class                                        = base.ThreeDCamera,
        ambient_camera_rotation                             = None,
        default_angled_camera_orientation_kwargs            = None,
        always_update_mobjects: bool                        = False,
        random_seed                                         = None,
        skip_animations:        bool                        = False,
        # Animation Args
        argsconfig:             Optional[AnimationConfig]   = None,
        ):
        # init args
        if animations is None:
            animations = []
        # constructor of base.scene
        super().__init__(renderer=renderer,
                         camera_class=camera_class,
                         always_update_mobjects=always_update_mobjects,
                         random_seed=random_seed,
                         skip_animations=skip_animations,
                         ambient_camera_rotation=ambient_camera_rotation,
                         default_angled_camera_orientation_kwargs=default_angled_camera_orientation_kwargs,
                         )
        # constructor of Animation
        super(base.ThreeDScene, self).__init__(None, 0, argsconfig=argsconfig)
        # Timeline instance is a Sequence of Animation
        self.__timeline_animations:    List[Animation] = animations
    @override
    def release_play_animation(self, scene:base.Scene):
        for animation in reversed(self.__timeline_animations):
            animation.release_play_animation(scene)
    @override
    def inject_play_animation(self, scene:base.Scene):
        for animation in self.__timeline_animations:
            animation.play_animation(scene)
        self.release_play_animation(self)
    @override
    def construct(self):
        self.play_animation(self)

    def add_animation(self, animation:Union[Animation, Sequence[Animation]]):
        if isinstance(animation, Animation):
            self.__timeline_animations.append(animation)
        elif isinstance(animation, Sequence):
            for index, item in enumerate(animation):
                if isinstance(item, Animation) is False:
                    raise ValueError(f"animations's item must be Animation, but index={index} is {type(item)}")
            self.__timeline_animations.extend(animation)
        else:
            raise ValueError(f"animation must be Animation or Sequence[Animation], but current is {type(animation)}")
        return self
    def pop_animation(self):
        return self.__timeline_animations.pop()
    def get_timeline_animations(self):
        return self.__timeline_animations
    def clear_animations(self):
        self.__timeline_animations.clear()
        return self

class VectorHelper(any_class):
    @property
    def up(self) -> np.ndarray:
        '''np.array((0.0, 1.0, 0.0))'''
        return base.UP
    @property
    def down(self) -> np.ndarray:
        '''np.array((0.0, -1.0, 0.0))'''
        return base.DOWN
    @property
    def left(self) -> np.ndarray:
        '''np.array((-1.0, 0.0, 0.0))'''
        return base.LEFT
    @property
    def right(self) -> np.ndarray:
        '''np.array((1.0, 0.0, 0.0))'''
        return base.RIGHT
    @property
    def front(self) -> np.ndarray:
        '''np.array((0.0, 0.0, 1.0))'''
        return base.OUT
    @property
    def back(self) -> np.ndarray:
        '''np.array((0.0, 0.0, -1.0))'''
        return base.IN
    @property
    def zero(self) -> np.ndarray:
        '''np.array((0.0, 0.0, 0.0))'''
        return base.ORIGIN

    @property
    def ul(self) -> np.ndarray:
        '''np.array((-1.0, 1.0, 0.0))'''
        return base.UL
    @property
    def ur(self) -> np.ndarray:
        '''np.array((1.0, 1.0, 0.0))'''
        return base.UR
    @property
    def dl(self) -> np.ndarray:
        '''np.array((-1.0, -1.0, 0.0))'''
        return base.DL
    @property
    def dr(self) -> np.ndarray:
        '''np.array((1.0, -1.0, 0.0))'''
        return base.DR

def set_animation_constructor(
    animations:     Union[Animation, Sequence[Animation]],
    constructor:    CDstructorType
    ):
    if isinstance(animations, Animation):
        animations.cd_constructor = constructor
    elif isinstance(animations, Sequence):
        for animation in animations:
            animation.cd_constructor = constructor
    else:
        raise ValueError(f"animations must be Animation or Sequence[Animation], but current is {type(animations)}")
def set_animation_destructor(
    animations:     Union[Animation, Sequence[Animation]],
    destructor:     CDstructorType
    ):
    if isinstance(animations, Animation):
        animations.cd_destructor = destructor
    elif isinstance(animations, Sequence):
        for animation in animations:
            animation.cd_destructor = destructor
    else:
        raise ValueError(f"animations must be Animation or Sequence[Animation], but current is {type(animations)}")

if __name__ == "__main__":
    pass





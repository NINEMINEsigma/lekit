from lekit.Internal import *
from lekit.Visual.Manim import *

class UncreateDestructor(invoke_callable):
    def __init__(self, duration=0.36):
        self.duration = duration
    def invoke(self, current:Animation, scene:base.Scene):
        Animation(current.ref_value, self.duration,
                  argsconfig=AnimationConfig(animate_creater=base.Uncreate)
            ).play_animation(scene)

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

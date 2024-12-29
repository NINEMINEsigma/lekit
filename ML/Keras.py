from typing                     import *
from lekit.File.Core            import (
                                       tool_file,
    UnWrapper                   as     UnWrapper2Str,
    Wrapper                     as     Wrapper2File
)
from lekit.Math.Core            import *
from keras.api.models import (
    Sequential                  as     KerasSequentialModel,
    load_model                  as     load_keras_model,
    save_model                  as     save_keras_model,
    clone_model                 as     clone_keras_model,
    Model                       as     KerasBaseModel,
    model_from_json             as     load_keras_model_from_json
)
from lekit.ML.KerasInternal.VerboseType import *
from lekit.ML.KerasInternal.LossType    import *

from tensorflow                 import Tensor
from keras.api.callbacks        import History as KerasHistory
from keras.api.optimizers       import Optimizer
from keras.api                  import losses, metrics
from keras.api.layers           import Layer
from keras.api.losses           import Loss
from keras.api.metrics          import Metric
from keras.api.callbacks        import Callback as KerasCallback
from keras.api.initializers     import Initializer

nparray_or_tensor = Union[np.ndarray, Tensor]

class light_keras_sequential:
    def __init__(
        self,
        initconfig:     Optional[Any]               = None,
        *,
        initmodel:      Optional[KerasSequentialModel] = None,
        initfile:       Optional[Union[str, tool_file]] = None,
        initlayers:     Optional[Sequence[Layer]]   = None,
        initdict_json:  dict                        = None,
        # initconfig = init*...
        trainable:      bool                        = True,
        name:           Optional[str]               = None
        ):
        self.last_result: Any  = None
        self.model:     KerasSequentialModel = None
        if initconfig is None:
            initconfig = self._init_load_first_item(
                initmodel,
                initfile,
                # not insert layers, see <if initconfig is None>
                initdict_json,
            )
        if initconfig is None:
            self.load(KerasSequentialModel(layers=initlayers ,trainable=trainable, name=name))
        else:
            self.load(initconfig)

    def _init_load_first_item(self, *args):
        for item in args:
            if item is not None:
                return item
        return None

    def load(
        self,
        initconfig:     Union[
            KerasSequentialModel,
            str,
            tool_file,
            dict,
            Sequence[Layer]
            ]
        ) -> Self:
        if isinstance(initconfig, KerasSequentialModel):
            self.model = initconfig
        elif isinstance(initconfig, (str, tool_file)):
            target = tool_file(UnWrapper2Str(initconfig))
            if target.get_extension() == "json":
                target.open('r', True)
                self.model = load_keras_model_from_json(target.data)
            elif (target.get_extension() == "h5" or
                  target.get_extension() == "keras" or
                  target.get_extension() == "weights"
            ):
                self.model = load_keras_model(UnWrapper2Str(target))
            else:
                print(f"target file <{UnWrapper2Str(target)}> with {target.get_extension()} is maybe not supported")
                self.model = load_keras_model(UnWrapper2Str(target))
        elif isinstance(initconfig, dict):
            self.model = load_keras_model_from_json(initconfig)
        elif isinstance(initconfig, Sequence):
            self.model = KerasSequentialModel(initconfig)
        else:
            raise ValueError(f"initconfig type {type(initconfig)} is not supported")
        return self
    def save(self, file:Union[str, tool_file]):
        file = Wrapper2File(file)
        if file.in_extensions("keras"):
            self.model.save(UnWrapper2Str(file))
        elif file.in_extensions("weights", "h5"):
            self.model.save_weights(UnWrapper2Str(file))
        else:
            self.model.save(UnWrapper2Str(file))

    def add(
        self,
        layer:          Union[Layer, Sequence[Layer]],
        rebuild:        bool    = True
        ):
        if isinstance(layer, Layer):
            self.model.add(layer, rebuild)
        elif isinstance(layer, Sequence):
            for item in layer:
                self.model.add(item, rebuild)
        return self
    def pop(
        self,
        rebuild:        bool    = True,
        time:           int     = 1
        ):
        for _ in range(time):
            self.model.pop(rebuild)
        return self
    def compile(
        self,
        name_or_instance_of_optimizer:  Union[Optimizer, str]               = "rmsprop",
        name_or_instance_of_loss:       Optional[Union[str, losses_type]]   = None,
        **kwargs
        ):
        self.model.compile(
            optimizer=name_or_instance_of_optimizer,
            loss=name_or_instance_of_loss,
            **kwargs)
        return self
    def fit[_X, _Y](
        self,
        train_or_trains_of_dataX:                _X = None,
        label_or_labels_or_dataY:                _Y = None,
        batch_size:                   Optional[int] = None, #default to 32
        epochs:                                 int = 1,
        verbose:                       verbose_type = "auto",
        callbacks:              List[KerasCallback] = None,
        validation_split:           NumberBetween01 = 0.0,
        validation_data:              Tuple[_X, _Y] = None,
        shuffle:                               bool = True,
        class_weight:    Optional[Dict[int, float]] = None,
        sample_weight:  Optional[nparray_or_tensor] = None,
        initial_epoch:                          int = 0,
        steps_per_epoch:              Optional[int] = None,
        validation_steps:             Optional[int] = None,
        validation_batch_size:        Optional[int] = None,
        validation_freq:                        int = 1,
    ) -> KerasHistory:
        return self.model.fit(
                                x = train_or_trains_of_dataX,
                                y = label_or_labels_or_dataY,
                           epochs = epochs,
                       batch_size = batch_size,
                          verbose = verbose,
                        callbacks = callbacks,
                 validation_split = validation_split,
                  validation_data = validation_data,
                          shuffle = shuffle,
                     class_weight = class_weight,
                    sample_weight = sample_weight,
                    initial_epoch = initial_epoch,
                  steps_per_epoch = steps_per_epoch,
                 validation_steps = validation_steps,
            validation_batch_size = validation_batch_size,
                  validation_freq = validation_freq
                  )
    def evaluate[_X, _Y](
        self,
        dataX:          _X                          = None,
        dataY:          _Y                          = None,
        batch_size:     Optional[int]               = None,
        verbose:        verbose_type                = "auto",
        sample_weight:  Optional[nparray_or_tensor] = None,
        steps:          Optional[int]               = None,
        callbacks:      List[KerasCallback]         = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.model.evaluate(
            x = dataX,
            y = dataY,
            batch_size = batch_size,
            verbose = verbose,
            sample_weight = sample_weight,
            steps = steps,
            callbacks = callbacks,
            return_dict = True,
            **kwargs
        )
        return self
    def predict[_X](
        self,
        dataX:          _X                          = None,
        batch_size:     Optional[int]               = None,
        verbose:        verbose_type                = "auto",
        steps:          Optional[int]               = None,
        callbacks:      List[KerasCallback]         = None,
        **kwargs,
    ) -> np.ndarray:
        return self.model.predict(
            x = dataX,
            batch_size = batch_size,
            verbose = verbose,
            steps = steps,
            callbacks = callbacks,
            **kwargs
        )
    def predict_classes[_X](
        self,
        dataX:          _X                          = None,
        batch_size:     Optional[int]               = None,
        verbose:        verbose_type                = "auto",
        steps:          Optional[int]               = None,
        callbacks:      List[KerasCallback]         = None,
        **kwargs,
    ) -> np.ndarray:
        #TODO
        return np.argmax(self.model.predict(
            x = dataX,
            batch_size = batch_size,
            verbose = verbose,
            steps = steps,
            callbacks = callbacks,
            **kwargs
        ), axis=1)

if __name__ == "__main__":
    pass

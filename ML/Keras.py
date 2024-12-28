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
from keras.api.optimizers       import Optimizer
from keras.api                  import losses, metrics
from keras.api.layers           import Layer
from keras.api.losses           import Loss
from keras.api.metrics          import Metric
from keras.api.callbacks        import Callback as KerasCallback
from keras.api.initializers     import Initializer

losses_type     = TypeVar('losses_type', Union[
    Loss,
    Union[
    losses.CTC,
    losses.BinaryCrossentropy,
    losses.BinaryFocalCrossentropy,
    losses.CategoricalCrossentropy,
    losses.CategoricalFocalCrossentropy,
    losses.CategoricalHinge,
    losses.Circle,
    losses.CosineSimilarity,
    losses.Dice,
    losses.Hinge,
    losses.Huber,
    losses.KLDivergence,
    losses.LogCosh,
    losses.MeanAbsoluteError,
    losses.MeanAbsolutePercentageError,
    losses.MeanSquaredError,
    losses.MeanSquaredLogarithmicError,
    losses.Poisson,
    losses.SparseCategoricalCrossentropy,
    losses.SquaredHinge,
    losses.Tversky,
    losses.binary_crossentropy,
    losses.binary_focal_crossentropy,
    losses.categorical_crossentropy,
    losses.categorical_focal_crossentropy,
    losses.categorical_hinge,
    losses.circle,
    losses.cosine_similarity,
    losses.ctc,
    losses.dice,
    losses.hinge,
    losses.huber,
    losses.kl_divergence,
    losses.log_cosh,
    losses.mean_absolute_error,
    losses.mean_absolute_percentage_error,
    losses.mean_squared_error,
    losses.mean_squared_logarithmic_error,
    losses.poisson,
    losses.sparse_categorical_crossentropy,
    losses.squared_hinge,
    losses.tversky,
    ],
    Callable[[Any, Any], Any]
    ])
'''
loss:
        Loss function. May be a string (name of loss function), or
        a `keras.losses.Loss` instance. See `keras.losses`. A
        loss function is any callable with the signature
        `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth
        values, and `y_pred` are the model's predictions.
        `y_true` should have shape `(batch_size, d0, .. dN)`
        (except in the case of sparse loss functions such as
        sparse categorical crossentropy which expects integer arrays of
        shape `(batch_size, d0, .. dN-1)`).
        `y_pred` should have shape `(batch_size, d0, .. dN)`.
        The loss function should return a float tensor.
'''
verbose_auto                = Literal["auto"]
verbose_silent              = Literal[0]
verbose_progress_bar        = Literal[1]
verbose_one_line_per_epoch  = Literal[2]

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
        layer:          Layer,
        rebuild:        bool    = True
        ):
        self.model.add(layer, rebuild)
        return self
    def pop(
        self,
        rebuild:        bool    = True
        ):
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
    def fit(
        self,
        train_or_trains_of_dataX                    = None,
        label_or_labels_or_dataY                    = None,
        batch_size:                   Optional[int] = None, #default to 32
        epochs:                                 int = 1,
        verbose:           Literal["auto", 0, 1, 2] = "auto",
        callbacks:              List[KerasCallback] = None,
        validation_split:           NumberBetween01 = 0.0,
        validation_data:                      tuple = None,
        shuffle:                               bool =True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self.last_result = self.model.fit(
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
        return self
    def evaluate(
        self,
        dataX=None,
        DataY=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        self.model.evaluate(dataX, DataY, batch_size, verbose, callbacks)
        return self

if __name__ == "__main__":
    pass

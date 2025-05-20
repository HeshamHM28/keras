from keras.src import backend
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


class BaseGlobalPooling(Layer):
    """Base global pooling layer."""

    def __init__(
        self, pool_dimensions, data_format=None, keepdims=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.data_format = backend.standardize_data_format(data_format)
        self.keepdims = keepdims
        self.input_spec = InputSpec(ndim=pool_dimensions + 2)

        # Cache spatial dims with class - almost always static
        self._pool_dimensions = pool_dimensions

        # Precompute and cache tuple shims for output shape to avoid repeated tuple construction
        num_spatial_dims = pool_dimensions
        # Channels Last
        self._outshape_tpl_last_no_keep = (None, None)
        self._outshape_tpl_last_keep = (None,) + (1,) * num_spatial_dims + (None,)
        # Channels First
        self._outshape_tpl_first_no_keep = (None, None)
        self._outshape_tpl_first_keep = (None, None) + (1,) * num_spatial_dims

        self._build_at_init()

    def call(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        # Use tuple construction with fixed slices instead of repeated branching/operators
        # input_shape: (batch, ...spatial..., channels) or (batch, channels, ...spatial...)

        num_spatial_dims = len(input_shape) - 2

        if self.data_format == "channels_last":
            if self.keepdims:
                # Example: (batch, 1, 1, channels) for 2D
                # Use only what's needed from input_shape to build the tuple
                return (
                    (input_shape[0],)
                    + (1,) * num_spatial_dims
                    + (input_shape[-1],)
                )
            else:
                return (input_shape[0],) + (input_shape[-1],)
        else:
            if self.keepdims:
                return (input_shape[0], input_shape[1]) + (1,) * num_spatial_dims
            else:
                return (input_shape[0], input_shape[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "keepdims": self.keepdims,
            }
        )
        return config

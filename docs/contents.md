# ConvNetJS Documentation

## [Nets](./nets/contents.md)

### [`MagicNet`](./nets/magicnet.md)

The `MagicNet` class 

### [`Net`](./nets/net.md)

A `Net` is essentially a set of layers. Passing a `Vol` forward through the `Net` passes it through each of the `Net`'s layers.

### [`Brain`](./nets/brain.md)

A `Brain` is a 

## [Layers](./layers/contents.md)

A neural network is composed of a series of layers.

### [`Layer`](./layers/layer.md)

`Layer` is a generic class for layers. All layers extend from this class.

#### [`Layer.fromJSON`](./layers/layer.md#fromJSON)

The `fromJSON` method creates a new `Layer` from a JSON serialisation of a layer.

### [`ConvLayer`](./layers/convlayer.md)

`ConvLayer` extends `Layer` and implements a [Convolutional layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer).

### [`DropoutLayer`](./layers/dropoutlayer.md)

`DropoutLayer` extends `Layer` and implements a [Dropout layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout_.22layer.22). 

### [`FullyConnLayer`](./layers/fullyconnlayer.md)

`FullyConnLayer` extends `Layer` and implements a Fully Connected layer.

### [`InputLayer`](./layers/inputlayer.md)

`InputLayer` extends `Layer` and implements a dummy layer that declares the size of the input Vols to a network.

### [`MaxoutLayer`](./layers/maxoutlayer.md)

`MaxoutLayer` extends `Layer` and implements a []()

### [`PoolLayer`](./layers/poollayer.md)

`PoolLayer` extends `Layer` and implements a [pooling layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer).

### [`RegressionLayer`](./layers/regressionlayer.md)

`RegressionLayer` extends `Layer` and implements a []().

### [`ReluLayer`](./layers/relulayer.md)

`ReluLayer` extends `Layer` and implements a [rectified linear units layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#ReLU_layer).

### [`SigmoidLayer`](./layers/sigmoidlayer.md)

`SigmoidLayer` extends `Layer` and implements a 

### [`SoftMaxLayer`](./softmaxlayer.md)

`SoftMaxLayer` extends `Layer`.

### [`SVMLayer`](./svmlayer.md)

`SVMLayer` extends `Layer`.

### [`TanhLayer`](./layers/tanhlayer.md)

`TanhLayer` extends `Layer`. 

## [Structures](./structures/contents.md)

### [`VolType`](./structures/voltype.md)

`VolType` is a constructor that creates a new Typed Object based on the dimensions for a Vol.

#### [`VolType#fromJSON`](./structures/voltype.md#voltypefromjson)

Creates a new Vol of a particular type from a JSON representation.

### [`%Vol%`](./structured/vol.md)

A Vol represents a three-dimensional array of 64-bit floats.

#### [`%Vol%.sx`](./structures/vol.md#volsx)

The `sx` property returns the size of the 'width' of the Vol.

#### [`%Vol%.sy`](./structures/vol.md#volsy)

The `sy` property returns the size of the 'height' of the Vol.

#### [`%Vol%.depth`](./structures/vol.md#depth)

The `depth` property return the size of the 'depth' of the Vol.

#### [`%Vol%.w`](./structured/vol.md#w)

The `w` property is a three dimensional 64-bit float array.

#### [`%Vol%.dw`](./structured/vol.md#dw)

The `dw` property is a three dimensional 64-bit float array.
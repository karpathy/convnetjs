# ConvNetJS Documentation

## Nets

### `MagicNet`

### `Net`

### `Brain`

## Layers

### `Layer`

Generic class for layers. 

#### [`Layer.fromJSON`](./layers/layer.md#fromJSON)

The `fromJSON` 

### [`ConvLayer`](./layers/convlayer.md)

`ConvLayer` extends `Layer` and inplements a ...

### `DropoutLayer`

### `FullyConnLayer`

### `InputLayer`

### `MaxoutLayer`

### `PoolLayer`

### `RegressionLayer`

### `ReluLayer`

### `SigmoidLayer`

### `SoftMaxLayer`

### `SVMLayer`

### `TanhLayer`


## Vols

### `VolType`

`VolType` is a constructor that creates a new Typed Object based on the dimensions for a Vol.

#### `VolType#fromJSON`

Creates a new Vol of a particular type from a JSON representation.

### `%Vol%`

A Vol represents a three-dimensional array of 64-bit floats.

#### [`%Vol%.sx`](./structures/vol.md#sx)

The `sx` property returns the size of the 'width' of the Vol.

#### [`%Vol%.sy`](./structures/vol.md#sy)

The `sy` property returns the size of the 'height' of the Vol.

#### [`%Vol%.depth`](./structures/vol.md#depth)

The `depth` property return the size of the 'depth' of the Vol.

#### [`%Vol%.w`](./structured/vol.md#w)

The `w` property is a three dimensional 64-bit float array.

#### [`%Vol%.dw`](./structured/vol.md#dw)

The `dw` property is a three dimensional 64-bit float array.
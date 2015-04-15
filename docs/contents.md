# ConvNetJS Documentation

## Nets

### `MagicNet`

### `Net`

### `Brain`

## Layers

### `Layer`

Generic class for layers. 

### `ConvLayer`

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

### `Vol`

#### `Vol#get`

#### `Vol#set`

#### `Vol#add`

#### `Vol#getGrad`

#### `Vol#setGrad`

#### `Vol#addGrad`

#### `Vol#cloneAndZero`

#### `Vol#clone`

Clones a `Vol`.

#### `Vol#addFrom`

#### `Vol#addFromScaled`

#### `Vol#setConst`

#### `Vol#toJSON`

Creates a JSON representation of a `Vol`.

### `Vol.fromImage`

Creates a new `Vol` from an `<img>` element. This won't work in a Web Worker or in Node.js as it requires the DOM.

### `Vol.fromImageData`

Creates a new `Vol` from an `ImageData` object, such as from a `<canvas>` element.

### [`Vol.fromJSON`](./)

Creates a new `Vol` from a JSON object or string.
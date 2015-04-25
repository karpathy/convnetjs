# Net

### `new Net(layers)`

The `Net` class create a new 

##### Parameters

* **`layers`** - a one dimensional array of layers to make up the network.

##### Returns

A new `Net` will be returned. 

### `Net#forward(V, is_training = false)

Use this method to pass a `Vol` forward through a network.

##### Parameters

* **`V`** - A [`Vol`](../structures/vol.md) to pass forward through the network. The size of the Vol should be the same size as the [`InputLayer`](../layers/inputlayer.md) for the `Net`.
* **`is_training`** - This param is used internally by a [`Trainer`](../trainers/trainer.md) to indicate to the `Net` that it is being trained.

##### Returns

This method returns the result of passing the `Vol` specified as the first parameter, `V`, through the network. 

### `Net.fromJSON(json)`

This method create a new `Net` from a 

##### Parameters

* **`json`** - Either a plain JavaScript object or a JSON string. If `json` is a string, it will be decoded using `JSON.parse()`.

##### Returns

A new `Net` will be returned. The `VolType` that the `Vol` returned is an instance of is available as the `constructor` property of the `Vol`.
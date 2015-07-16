# ConvLayer

### `new ConvLayer({filters, sx, in_depth, in_sx, in_sy, })`

### `ConvLayer.prototype.forward(V, is_training = false)`

**Parameters:**
* **`V`** - A Vol to pass forward through the network.
* **`is_training`** - Optional boolean argument to tell layer the network is being trained.

**Returns:**
A new Vol.
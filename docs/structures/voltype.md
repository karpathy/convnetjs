# `VolType`

`VolType` is a class used to create constructors for creating Vols. `VolType` internally uses the `TypedObject.StructType` constructor to create VolTypes. This means that the Vols have a predefined size, which allows the storage behind a Vol to be exposed as an `ArrayBuffer` (and thus transferred easily to a Web Worker, or manipulating using a Typed Array) and for a JavaScript to better optimise Vols.

### `new VolType(sx = 1, sy = 1, depth = 1)`

The VolType constructor returns a new instance of `VolType`, which can be used to create Vols of the specified size.

For example, if you are going to be using Vols to represent 32 by 32  RGBA pixels, you would use the `VolType` constructor to create a new VolType with sx = 32, sy = 32 and depth = 4. 

##### Parameters

 * **`sx`** - The size of the first dimension of the vols to be made using this constructor. Defaults to 1.
 * **`sy`** 

##### Returns

A constructor will be returned, which can be used to create new Vols of the size `sx * sy * depth`. 

### `VolType.fromJSON(json)`

This method takes a JSON encoded string or a plain object

##### Parameters

 * **`json`** -  Either a plain JavaScript object or a JSON string. If `json` is a string, it will be decoded using `JSON.parse()`.

##### Returns

A new [`Vol`](./vol.md) will be returned. The `VolType` that the `Vol` returned is an instance of is available as the `constructor` property of the `Vol`. 

##### Example usage:

The following creates an object representing a `Vol`, and a 

```javascript
// As the JSON object has no .w or .dw property, these will be initialised to 0.
let myVolObject = {sx:32,sy:32,depth:4};
let myVolString = JSON.stringify(myVolObject);
let firstVol = VolType.fromJSON(myVolObject);
let secondVol = VolType.fromJSON(myVolString);
```
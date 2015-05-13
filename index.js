(function() {

	// Cover the various ways TypedObjects are specified.
	if(typeof StructType == undefined){
		if(typeof TypedObject == undefined){
			//TypedObject = require('typed-objects')();
		}
		Any = TypedObject.Any;
		ArrayType = TypedObject.ArrayType; 
		StructType = TypedObject.StructType;
		float32 = TypedObject.float32;
		float64 = TypedObject.float64;
		int16 = TypedObject.int16;
		int32 = TypedObject.int32;
		int8 = TypedObject.int8;
		objectType = TypedObject.objectType
		storage = TypedObject.storage;
		string = TypedObject.string;
		uint16 = TypedObject.uint16;
		uint32 = TypedObject.uint32;
		uint8 = TypedObject.uint8;
		uint8Clamped = TypedObject.uint8Clamped;
	}

	if(typeof SIMD == undefined){
		// require('simd');
	}

	if (typeof module === "undefined" || typeof module.exports === "undefined") {
		// in ordinary browser attach library to window
		window.convnetjs = require('./src/index.js'); 
	} else {
		// in commonjs
		module.exports = require('./src/index.js'); 
	}
})();

(function() {

	require('babel/polyfill');

	// Cover the various ways TypedObjects are specified.
	if(typeof TypedObject !== 'undefined'){
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
	} else {
		require('typed-objects')()
	}

	if(typeof SIMD == undefined){
		require('./external/simd.js');
	}

	if (typeof module === "undefined" || typeof module.exports === "undefined") {
		// in ordinary browser attach library to window
		window.convnetjs = require('./lib/index.js'); 
	} else {
		// in commonjs
		module.exports = require('./lib/index.js'); 
	}
})();

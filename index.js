(function() {

	require('babel/polyfill');

	if(typeof TypedObject === 'undefined'){
		TypedObject = require('typed-objects');
	}

	require('simd');

	if (typeof module === "undefined" || typeof module.exports === "undefined") {
		// in ordinary browser attach library to window
		window.convnetjs = require('./lib/index.js'); 
	} else {
		// in commonjs
		module.exports = require('./lib/index.js'); 
	}
})();

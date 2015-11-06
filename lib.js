"use strict";
var lib = require('./build/convnetjs');
lib.netprototype = require('./src/network/prime_net');

if(typeof window !== "undefined"){
	window.convnetjs = lib; // in ordinary browser attach library to window
}

if (typeof module !== "undefined" || typeof module.exports !== "undefined") {
	module.exports = lib; // in nodejs
}
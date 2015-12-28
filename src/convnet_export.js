(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.convnetjs = lib; // in ordinary browser attach library to window
  } else {
  	// for nodejs, call by var deepqlearn = convnetjs.deepqlearn()
  	lib.deepqlearn = function() {
      return require(__dirname+'/deepqlearn.js');
    };
    module.exports = lib; // in nodejs
  }
})(convnetjs);

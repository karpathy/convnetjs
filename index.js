(function() {
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.convnetjs = require('./src/index.js'); // in ordinary browser attach library to window
  } else {
    module.exports = require('./src/index.js'); // in commonjs
  }
})();

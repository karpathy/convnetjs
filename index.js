(function(Math) {
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.convnetjs = require('./src/index.js'); // in ordinary browser attach library to window
  } else {
    module.exports = require('./src/index.js'); // in commonjs
  }
//If you pass globals with long names here, they can be compressed better by Uglify.
})(Math);

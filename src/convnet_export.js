(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    // By declaring convnetjs globally, we have already made it available.
    // Nothing to do here.
  } else {
    module.exports = lib; // in nodejs
  }
})(convnetjs);

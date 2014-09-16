
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  var getopt = global.getopt;

  var InputLayer = function(opt) {
    var opt = opt || {};

    // required: depth
    this.out_depth = getopt(opt, ['out_depth', 'depth'], 0);

    // optional: default these dimensions to 1
    this.out_sx = getopt(opt, ['out_sx', 'sx', 'width'], 1);
    this.out_sy = getopt(opt, ['out_sy', 'sy', 'height'], 1);
    
    // computed
    this.layer_type = 'input';
  }
  InputLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act = V;
      return this.out_act; // simply identity function for now
    },
    backward: function() { },
    getParamsAndGrads: function() {
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
    }
  }

  global.InputLayer = InputLayer;
})(convnetjs);

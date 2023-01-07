(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  var FourierFeatureLayer = function(opt) {

    var opt = opt || {};

    // required
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional

    // computed
    this.out_depth = this.in_depth * 2;
    this.out_sx = this.in_sx
    this.out_sy = this.in_sy;
    this.layer_type = 'fourier_feature';

    // TODO - probably we don't need switchx and switchy in this class anymore?
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
  }

  FourierFeatureLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      var n=0; // a counter for switches
      for(var d=0;d<this.out_depth / 2; d++) {
        var x = 0;
        var y = 0;
        for(var ax=0; ax<this.out_sx; x+=1,ax++) {
          for(var ay=0; ay<this.out_sy; y+=1,ay++) {

            // for the first "half" of the fourier feature - use sine
            var v = V.get(ax, ay, d);
            var a = Math.sin(2 * Math.PI * v);
            n++;

            A.set(ax, ay, d, a);
          }
        }
      }
      for(var d=0;d<this.out_depth / 2; d++) {
        for(var ax=0; ax<this.out_sx; x+=1,ax++) {
          for(var ay=0; ay<this.out_sy; y+=1,ay++) {

            // for the first "half" of the fourier feature - use cosine
            var v = V.get(ax, ay, d);
            var a = Math.cos(2 * Math.PI * v);
            n++;

            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() { 
      // pooling layers have no parameters, so simply compute 
      // gradient wrt data here
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      var A = this.out_act; // computed in forward pass 

      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var x = 0;
        var y = 0;
        for(var ax=0; ax<this.out_sx; x+=1,ax++) {
          for(var ay=0; ay<this.out_sy; y+=1,ay++) {

            var chain_grad = this.out_act.get_grad(ax,ay,d);
            V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
            n++;

          }
        }
      }
    },
    getParamsAndGrads: function() {
      return [];
    },
    toJSON: function() {
      var json = {};
      json.in_depth = this.in_depth;
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
      this.in_depth = json.in_depth;
      this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth); // need to re-init these appropriately
      this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    }
  }

  global.FourierFeatureLayer = FourierFeatureLayer;

})(convnetjs);

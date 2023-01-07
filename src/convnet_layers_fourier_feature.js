(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  var FourierFeatureLayer = function(opt) {
    /**
     * Based on the paper "Fourier Features Let Networks Learn
     * High Frequency Functions in Low Dimensional Domains" (2020) presented 
     * at NeurIPS (https://bmild.github.io/fourfeat/index.html) 
     * by Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil,
     * Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron and Ren Ng.
     * 
     * Please see the Python implementation to see examples of the concept in action:
     *  https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb. 
     */

    var opt = opt || {};

    // required
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional - whether to factor in a random number (sampled from a Gaussian) in the mapping or not (defaults to -1, which signifies not to include it)
    this.gaussianMappingScale = typeof opt.gaussian_mapping_scale !== 'undefined' ? opt.gaussian_mapping_scale : -1;

    // computed
    this.out_depth = this.in_depth * 2 * Math.abs(this.gaussianMappingScale);
    this.out_sx = this.in_sx
    this.out_sy = this.in_sy;
    this.layer_type = 'fourier_feature';

  }

  FourierFeatureLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var mappedFeature = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      for(var d=0;d<this.out_depth; d++) {
        var x = 0;
        var y = 0;
        for(var ax=0; ax<this.out_sx; x+=1,ax++) {
          for(var ay=0; ay<this.out_sy; y+=1,ay++) {
            var v = V.get(ax, ay, d % this.in_depth);
            var randomProjFactor = 1;
            if (this.gaussianMappingScale > 0) {
              randomProjFactor *= global.randn(0.0, 1.0);
            }
            var a;
            // for the first "half" of the fourier feature - use sine
            if (d < (this.out_depth / 2)) {
              a = Math.cos(2 * Math.PI * v * randomProjFactor);
            } else {
            // use sine for the second "half"
              a = Math.sin(2 * Math.PI * v * randomProjFactor);
            }
            mappedFeature.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = mappedFeature;
      return this.out_act;
    },
    backward: function() { 
      // no parameters, so simply compute gradient wrt data here
      this.in_act.dw = global.zeros(this.in_act.w.length); // zero out gradient wrt data
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
    }
  }

  global.FourierFeatureLayer = FourierFeatureLayer;

})(convnetjs);

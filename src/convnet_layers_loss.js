(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  // Customized Layers=============================
  var BinaryReinforceLayer = function(opt) {
    var opt = opt || {};
    
    // optional
    this.min_val = typeof opt.min_val !== 'undefined' ? opt.min_val : 0;
    this.max_val = typeof opt.max_val !== 'undefined' ? opt.max_val : 1.0;
    this.threshold = typeof opt.threshold !== 'undefined' ? opt.threshold : 0.3; //arbitrary magic number
    
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'binaryReinforce';
    this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    
    this.init();
  }
  
  BinaryReinforceLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    },
    
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act.setConst(0.0);
      for(var i = 0; i < this.in_act.w.length; i++){
        this.out_act.w[i] = (this.in_act.w[i] < this.threshold) ? this.min_val : this.max_val;
      }
      
      return this.out_act; // identity function
    },
    
    backward: function(y) {
      //clear gradients
      this.in_act.setGradConst(0.0);
      var cost = 0.0;
      var N = this.in_act.w.length;
      var indicator;
      // compute and accumulate gradient wrt weights and bias of this layer
      if(!y.length){
        //y is a scaler label
        for(var i=0;i<N;i++) {
          indicator = i === y? this.max_val : this.min_val;
          //ideal output = max_val, dw should be negative
          //ideal output = min_val, dw should be positive
          if(indicator == this.max_val || this.out_act.w[i] == this.max_val){
            //it is a fired neuron
            this.in_act.dw[i] = (this.out_act.w[i] - indicator);
          }
          if(this.in_act.dw[i] != 0){
            cost++;
          }
        }
      }else{
        //y is a volume
        for(var i=0;i<N;i++) {
          indicator = (y[i] < this.threshold) ? this.min_val : this.max_val;
          if(indicator == this.max_val || this.out_act.w[i] == this.max_val){
            //it is a fired neuron
            this.in_act.dw[i] = (this.out_act.w[i] - indicator);
          }
          if(this.in_act.dw[i] != 0){
            cost++;
          }
        }
      }
      
      return cost;
    },
    getParamsAndGrads: function() { 
      return [];
    },
    toJSON: function() {
      var json = {};
      json.min_val = this.min_val;
      json.max_val = this.max_val;
      json.threshold = this.threshold;
      
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      return json;
    },
    fromJSON: function(json) {
      this.min_val = json.min_val;
      this.max_val = json.max_val;
      this.threshold = json.threshold || 0.3;
      
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      
      this.init();
    }
  }
  
  
  //===============================================
  // Layers that implement a loss. Currently these are the layers that 
  // can initiate a backward() pass. In future we probably want a more 
  // flexible system that can accomodate multiple losses to do multi-task
  // learning, and stuff like that. But for now, one of the layers in this
  // file must be the final layer in a Net.

  // This is a classifier, with N discrete classes from 0 to N-1
  // it gets a stream of N incoming numbers and computes the softmax
  // function (exponentiate and normalize to sum to 1 as probabilities should)
  var SoftmaxLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'softmax';
    
    this.init();
  }

  SoftmaxLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    },
    
    forward: function(V, is_training) {
      this.in_act = V;

      // compute max activation
      var as = V.w;
      var amax = V.w[0];
      for(var i=1;i<this.out_depth;i++) {
        if(as[i] > amax) amax = as[i];
      }

      // compute exponentials (carefully to not blow up)
      var es = global.zeros(this.out_depth);
      var esum = 0.0;
      for(var i=0;i<this.out_depth;i++) {
        var e = Math.exp(as[i] - amax);
        esum += e;
        es[i] = e;
      }

      // normalize and output to sum to one
      for(var i=0;i<this.out_depth;i++) {
        es[i] /= esum;
        this.out_act.w[i] = es[i];
      }

      this.es = es; // save these for backprop
      return this.out_act;
    },
    backward: function(y) {

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

      for(var i=0;i<this.out_depth;i++) {
        var indicator = i === y ? 1.0 : 0.0;
        var mul = -(indicator - this.es[i]);
        x.dw[i] = mul;
      }

      // loss is the class negative log likelihood
      return -Math.log(this.es[y]);
    },
    getParamsAndGrads: function() { 
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      
      this.init();
    }
  }

  // implements an L2 regression cost layer,
  // so penalizes \sum_i(||x_i - y_i||^2), where x is its input
  // and y is the user-provided array of "correct" values.
  var RegressionLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'regression';
    
    this.init();
  }

  RegressionLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    },
    
    forward: function(V, is_training) {
      this.in_act = V;
      for(var i = 0; i < this.in_act.w.length; i++){
        this.out_act.w[i] = this.in_act.w[i];
      }
      return this.out_act; // identity function
    },
    // y is a list here of size num_inputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to 
    // regress on dimension i and asking it to have value x
    backward: function(y) { 

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
      var loss = 0.0;
      if(y instanceof Array || y instanceof Float64Array) {
        for(var i=0;i<this.out_depth;i++) {
          var dy = x.w[i] - y[i];
          x.dw[i] = dy;
          loss += 0.5*dy*dy;
        }
      } else if(typeof y === 'number') {
        // lets hope that only one number is being regressed
        var dy = x.w[0] - y;
        x.dw[0] = dy;
        loss += 0.5*dy*dy;
      } else {
        // assume it is a struct with entries .dim and .val
        // and we pass gradient only along dimension dim to be equal to val
        var i = y.dim;
        var yi = y.val;
        var dy = x.w[i] - yi;
        x.dw[i] = dy;
        loss += 0.5*dy*dy;
      }
      return loss;
    },
    getParamsAndGrads: function() { 
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      
      this.init();
    }
  }

  var SVMLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'svm';
    
    this.init();
  }

  SVMLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    },
    
    forward: function(V, is_training) {
      this.in_act = V;
      for(var i = 0; i < this.in_act.w.length; i++){
        this.out_act.w[i] = this.in_act.w[i];
      }
      return this.out_act; // identity function
    },
    backward: function(y) {

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

      // we're using structured loss here, which means that the score
      // of the ground truth should be higher than the score of any other 
      // class, by a margin
      var yscore = x.w[y]; // score of ground truth
      var margin = 1.0;
      var loss = 0.0;
      for(var i=0;i<this.out_depth;i++) {
        if(y === i) { continue; }
        var ydiff = -yscore + x.w[i] + margin;
        if(ydiff > 0) {
          // violating dimension, apply loss
          x.dw[i] += 1;
          x.dw[y] -= 1;
          loss += ydiff;
        }
      }

      return loss;
    },
    getParamsAndGrads: function() { 
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      
      this.init();
    }
  }
  
  global.RegressionLayer = RegressionLayer;
  global.SoftmaxLayer = SoftmaxLayer;
  global.SVMLayer = SVMLayer;
  global.BinaryReinforceLayer = BinaryReinforceLayer;
})(convnetjs);


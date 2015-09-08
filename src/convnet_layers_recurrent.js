(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes: 
  // - RNNLayer is fully connected dot products with output feedback to its input
  // - LSTMLayer is long short term memory with memory block, input, forget, and output gates
  // putting them together in one file because they are very similar
  var LSTMLayer = function(opt) {
    var opt = opt || {};

    // required
    this.out_depth = opt.filters; //this is the num_output
    this.sx = 1; // filter size. Should be odd if possible, it's cleaner. output topology; TODO, support more than 1
    
    //1D input, the LSTM should not care about 2D topology
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    //this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'lstm';

    // initializations
    this.context = new Vol(1, 1, this.filters); // C
    this.prev_context = new Vol(1, 1, this.filters); // C
    this.contextH = new Vol(1, 1, this.filters); // tanh(C) -> to simplify bp computation
    
    this.filters = [];
    for(var i=0;i<this.out_depth;i++){
        var gates = [];
        for(var j = 0; j < 4; j++){
          //in, ig, fg, og
          gates.push(new Vol(1, 1, this.num_inputs));
        }
        this.filters.push(gates);
    };
    
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.biases = new Vol(4, 1, this.out_depth, bias); // 4 gate per unit
    
    // 4 gates per unit
    // w: the gate output AFTER transfer function
    // dw: the dw of the AFTER output after transfer function
    this.gateOut = new Vol(4, 1, this.out_depth, 0); 
    
    // 4 gates per unit
    // w: the gate output BEFORE transfer function
    // dw: the dw of the gate output BEFORE transfer function
    this.gateSum = new Vol(4, 1, this.out_depth, 0); 
  }
  
  function tanh(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  
  LSTMLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      var Vw = V.w;
      
      for(var i=0;i<this.out_depth;i++) {
        // compute the 4 gates
        // explicitly write out the name to avoid confusion, compromise code length
        var Xin = 0.0;
        var Xig = 0.0;
        var Xfg = 0.0;
        var Xog = 0.0;
        var wgates = this.filters[i];
        
        for(var d=0;d<this.num_inputs;d++) {
          Xin += Vw[d] * wgates[0].w[d];
          Xig += Vw[d] * wgates[1].w[d];
          Xfg += Vw[d] * wgates[2].w[d];
          Xog += Vw[d] * wgates[3].w[d];
        }
        
        Xin += this.biases.get(0,0,i);
        Xig += this.biases.get(1,0,i);
        Xfg += this.biases.get(2,0,i);
        Xog += this.biases.get(3,0,i);
        
        // set to gate memory
        this.gateSum.set(0,0,i, Xin);
        this.gateSum.set(1,0,i, Xig);
        this.gateSum.set(2,0,i, Xfg);
        this.gateSum.set(3,0,i, Xog);
        
        var Yin = tanh(Xin); // tanh
        var Yig = 1.0/(1.0+Math.exp(-Xig)); // sigmoid
        var Yfg = 1.0/(1.0+Math.exp(-Xfg)); // sigmoid
        var Yog = 1.0/(1.0+Math.exp(-Xog)); // sigmoid
        
        // set to gate memory
        this.gateOut.set(0,0,i, Yin);
        this.gateOut.set(1,0,i, Yig);
        this.gateOut.set(2,0,i, Yfg);
        this.gateOut.set(3,0,i, Yog);
        
        // update prev context
        var pre_C = this.context.w[i];
        this.prev_context.w[i] = pre_C;
        // compute new context
        this.context.w[i] = Yin * Yig + pre_C * Yfg;
        
        // compute the final output
        this.contextH.w[i] = tanh(this.context.w[i]);
        A.w[i] = this.contextH.w[i] * Yog;
      }
      
      this.out_act = A;
      return this.out_act;
    },
    
    //derivatives are copied from bp of nonlinearities.js
    backward: function() {
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(var i=0;i<this.out_depth;i++) {
        var wgates = this.filters[i];
        var chain_grad = this.out_act.dw[i];
        
        // dE/dYog and dE/dXog
        var dEdYog = this.contextH.w[i] * chain_grad;
        var Yog = this.gateOut.get(3,0,i);
        this.gateOut.set_grad(3,0,i, dEdYog);
        this.gateSum.set_grad(3,0,i, Yog * (1.0 - Yog) * dEdYog); // derivative of sigmoid 
        
        // loss after tanh(context)
        this.contextH.dw[i] = this.gateOut.get(3,0,i) * chain_grad;
        
        // calculate new loss into context
        var contextVal = this.context.w[i];
        var new_dEdcontext = (1 - contextVal * contextVal) * this.contextH.dw[i]; // derivative of tanh
        
        // update context loss
        this.prev_context.dw[i] = this.context.dw[i];
        this.context.dw[i] += new_dEdcontext * this.gateOut.get(2,0,i); //Yog * new_dEdcontext + prev_dEdcontext = dEdcontext
        
        // dE/dYfg
        var dEdYfg = this.prev_context.w[i] * this.context.dw[i];
        var Yfg = this.gateOut.get(2,0,i);
        this.gateOut.set_grad(2,0,i, dEdYfg);
        this.gateSum.set_grad(2,0,i, Yfg * (1.0 - Yfg) * dEdYfg); // derivative of sigmoid 
        
        // dE/dYig & dE/dYin
        var dEdYig = this.gateOut.get(0,0,i) * this.context.dw[i];
        var dEdYin = this.gateOut.get(1,0,i) * this.context.dw[i];
        var Yig = this.gateOut.get(1,0,i);
        var Yin = this.gateOut.get(0,0,i);
        
        this.gateOut.set_grad(1,0,i, dEdYig);
        this.gateOut.set_grad(0,0,i, dEdYin);
        this.gateSum.set_grad(1,0,i, Yig * (1.0 - Yig) * dEdYig); // derivative of sigmoid 
        this.gateSum.set_grad(0,0,i, (1.0 - Yin * Yin) * dEdYin); // derivative of tanh 
        
        for(var g = 0; g < 4; g++){
          for(var d=0;d<this.num_inputs;d++) {
            wgates[g].dw[d] += V.w[d] * this.gateSum.get_grad(g,0,i);// grad wrt params
            V.dw[d] +=  wgates[g].w[d] * this.gateSum.get_grad(g,0,i); // grad wrt input data
          }
          
          this.biases.set_grad(g,0,i, this.biases.get(g,0,i) +  this.gateSum.get_grad(g,0,i));
        }
      }
    },
    
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        for(var g=0; g<4; g++){
          response.push({params: this.filters[i][g].w, grads: this.filters[i][g].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
        }
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        // parameters for 4 gates
        var gateW = [];
        for(var g=0; g<4; g++){
          gateW = this.filters[i][g].toJSON();
        }
        json.filters.push(gateW);
      }
      json.biases = this.biases.toJSON();
      
      // memory blocks
      json.context = this.context.toJSON();
      json.prev_context = this.prev_context.toJSON();
      json.contextH = this.contextH.toJSON();
      json.gateOut = this.gateOut.toJSON();
      json.gateSum = this.gateSum.toJSON();
      
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var gateRaw = json.filters[i];
        var gateW = [];
        for(var g=0; g < gateRaw.length; g++){
          var v = new Vol(0,0,0,0);
          v.fromJSON(gateRaw[g]);
          gateW.push(v);
        }
        this.filters.push(gateW);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
      
      // memory blocks
      this.context = new Vol(0,0,0,0);
      this.context.fromJSON(json.context);
      
      this.prev_context = new Vol(0,0,0,0);
      this.prev_context.fromJSON(json.prev_context);
      
      this.contextH = new Vol(0,0,0,0);
      this.contextH.fromJSON(json.contextH);
      
      this.gateOut = new Vol(0,0,0,0);
      this.gateOut.fromJSON(json.gateOut);
      
      this.gateSum = new Vol(0,0,0,0);
      this.gateSum.fromJSON(json.gateSum);
    }
  }
  
  global.ConvLayer = LSTMLayer;
})(convnetjs);

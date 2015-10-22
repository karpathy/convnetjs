(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  var concat_vol = global.concat_vol;
  
  // lstm util function
  function tanh(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  
  function sigmoid(x){
    return 1.0/(1.0+Math.exp(-x));
  }
  
  // capping value
  function capValue(val, max, min){
      var valCap = (val > max)? max : val;
      valCap = (valCap < min)? min : valCap;
      return valCap;
  }

  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes: 
  // - LSTMLayer is long short term memory with memory block, input, forget, and output gates. The output are Not reconnected to the input
  // - LSTMRecurrentLayer is long short term memory with memory block, input, forget, and output gates.The output are reconnected to the input
  // putting them together in one file because they are very similar
  var LSTMRecurrentLayer = function(opt) {
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
    
    this.maxC = typeof opt.maxC !== 'undefined' ? opt.maxC : 10.0;
    this.maxCError = typeof opt.maxCError !== 'undefined' ? opt.maxCError : 10.0;
    this.maxHistoryStep = 5;
    
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.total_inputs = this.num_inputs + this.out_depth; //+this.out_depth
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'lstm-recurrent';

    // initializations
    this.reset();
    
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.biases = new Vol(4, 1, this.out_depth, bias); // 4 gate per unit
    
    this.filters = [];
    for(var i=0;i<this.out_depth;i++){
        this.filters.push(new Vol(4, 1, this.total_inputs)); //4 gates per cell
    };
    
    // 4 gates per unit
    // w: the gate output AFTER transfer function
    // dw: the dw of the AFTER output after transfer function
    this.gateOut = new Vol(4, 1, this.out_depth, 0.0); 
    
    // 4 gates per unit
    // w: the gate output BEFORE transfer function
    // dw: the dw of the gate output BEFORE transfer function
    this.gateSum = new Vol(4, 1, this.out_depth, 0.0); 
  }

  LSTMRecurrentLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = concat_vol(V,this.out_act);
      var A = new Vol(1, 1, this.out_depth, 0.0);
      var Vw = V.w;
      
      for(var i=0;i<this.out_depth;i++) {
        // compute the 4 gates
        // explicitly write out the name to avoid confusion, compromise code length
        var Xin = 0.0;
        var Xig = 0.0;
        var Xfg = 0.0;
        var Xog = 0.0;
       
        for(var d=0;d<this.num_inputs;d++) {
          Xin += Vw[d] * this.filters[i].get(0,0,d);
          Xig += Vw[d] * this.filters[i].get(1,0,d);
          Xfg += Vw[d] * this.filters[i].get(2,0,d);
          Xog += Vw[d] * this.filters[i].get(3,0,d);
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
        var Yig = sigmoid(Xig); // sigmoid
        var Yfg = sigmoid(Xfg); // sigmoid
        var Yog = sigmoid(Xog); // sigmoid
        
        // set to gate memory
        this.gateOut.set(0,0,i, Yin);
        this.gateOut.set(1,0,i, Yig);
        this.gateOut.set(2,0,i, Yfg);
        this.gateOut.set(3,0,i, Yog);
        
        // update prev context
        var pre_C = this.context.w[i];
        this.prev_context.w[i] = pre_C;
        
        // compute new context
        //resolve instability
        this.context.w[i] = Yin * Yig + pre_C * Yfg;
        this.context.w[i] = capValue(Yin * Yig + pre_C * Yfg, this.maxC, -this.maxC);
        
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
        var chain_grad = this.out_act.dw[i];
        
        // dE/dYog and dE/dXog
        var dEdYog = this.contextH.w[i] * chain_grad;
        var Yog = this.gateOut.get(3,0,i);
        this.gateOut.set_grad(3,0,i, dEdYog);
        this.gateSum.set_grad(3,0,i, Yog * (1.0 - Yog) * dEdYog); // derivative of sigmoid 
        
        // loss after tanh(context)
        this.contextH.dw[i] = this.gateOut.get(3,0,i) * chain_grad;
        
        // calculate new loss into context
        // Notice: this implementation is treating sequence length = 1
        var contextVal = this.context.w[i];
        var new_dEdcontext = (1 - contextVal * contextVal) * this.contextH.dw[i]; // derivative of tanh
        
        // update context loss
        this.prev_context.dw[i] = this.context.dw[i];
        
        //Yog * new_dEdcontext + prev_dEdcontext = dEdcontext, subject to cap
        //using cap function to prevent exploding gradient
        this.context.dw[i] = capValue(this.context.dw[i] + new_dEdcontext * this.gateOut.get(2,0,i), this.maxCError, -this.maxCError);
        
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
            this.filters[i].add_grad(g,0,d,V.w[d] * this.gateSum.get_grad(g,0,i));// grad wrt params
            V.dw[d] +=  this.filters[i].get(g,0,d) * this.gateSum.get_grad(g,0,i); // grad wrt input data
          }
          
          this.biases.set_grad(g,0,i, this.biases.get(g,0,i) +  this.gateSum.get_grad(g,0,i));
        }
      }
    },
    
    reset: function(){
      // initializations
      this.context = new Vol(1, 1, this.out_depth, 0.0); // C
      this.prev_context = new Vol(1, 1, this.out_depth, 0.0); // C
      this.contextH = new Vol(1, 1, this.out_depth, 0.0); // tanh(C) -> to simplify bp computation
      
      this.in_act_history = []; //stores x
      this.lambda_history = []; //stores X and h^-1
      this.gate_out_history = []; //stores Y and dE/dy (C)
    },
    
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
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
        json.filters.push(this.filters[i].toJSON());
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
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
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
    
    this.maxC = typeof opt.maxC !== 'undefined' ? opt.maxC : 10.0;
    this.maxCError = typeof opt.maxCError !== 'undefined' ? opt.maxCError : 10.0;

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'lstm';

    // initializations
    this.reset();
    
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.biases = new Vol(4, 1, this.out_depth, bias); // 4 gate per unit
    
    this.filters = [];
    for(var i=0;i<this.out_depth;i++){
        this.filters.push(new Vol(4, 1, this.num_inputs)); //4 gates per cell
    };
    
    // 4 gates per unit
    // w: the gate output AFTER transfer function
    // dw: the dw of the AFTER output after transfer function
    this.gateOut = new Vol(4, 1, this.out_depth, 0.0); 
    
    // 4 gates per unit
    // w: the gate output BEFORE transfer function
    // dw: the dw of the gate output BEFORE transfer function
    this.gateSum = new Vol(4, 1, this.out_depth, 0.0); 
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
       
        for(var d=0;d<this.num_inputs;d++) {
          Xin += Vw[d] * this.filters[i].get(0,0,d);
          Xig += Vw[d] * this.filters[i].get(1,0,d);
          Xfg += Vw[d] * this.filters[i].get(2,0,d);
          Xog += Vw[d] * this.filters[i].get(3,0,d);
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
        //resolve instability
        this.context.w[i] = Yin * Yig + pre_C * Yfg;
        this.context.w[i] = capValue(Yin * Yig + pre_C * Yfg, this.maxC, -this.maxC);
        
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
        var chain_grad = this.out_act.dw[i];
        
        // dE/dYog and dE/dXog
        var dEdYog = this.contextH.w[i] * chain_grad;
        var Yog = this.gateOut.get(3,0,i);
        this.gateOut.set_grad(3,0,i, dEdYog);
        this.gateSum.set_grad(3,0,i, Yog * (1.0 - Yog) * dEdYog); // derivative of sigmoid 
        
        // loss after tanh(context)
        this.contextH.dw[i] = this.gateOut.get(3,0,i) * chain_grad;
        
        // calculate new loss into context
        // Notice: this implementation is treating sequence length = 1
        var contextVal = this.context.w[i];
        var new_dEdcontext = (1 - contextVal * contextVal) * this.contextH.dw[i]; // derivative of tanh
        
        // update context loss
        this.prev_context.dw[i] = this.context.dw[i];
        
        //Yog * new_dEdcontext + prev_dEdcontext = dEdcontext, subject to cap
        //using cap function to prevent exploding gradient
        this.context.dw[i] = capValue(this.context.dw[i] + new_dEdcontext * this.gateOut.get(2,0,i), this.maxCError, -this.maxCError);
        
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
            this.filters[i].add_grad(g,0,d,V.w[d] * this.gateSum.get_grad(g,0,i));// grad wrt params
            V.dw[d] +=  this.filters[i].get(g,0,d) * this.gateSum.get_grad(g,0,i); // grad wrt input data
          }
          
          this.biases.set_grad(g,0,i, this.biases.get(g,0,i) +  this.gateSum.get_grad(g,0,i));
        }
      }
    },
    
    reset: function(){
      // initializations
      this.context = new Vol(1, 1, this.out_depth, 0.0); // C
      this.prev_context = new Vol(1, 1, this.out_depth, 0.0); // C
      this.contextH = new Vol(1, 1, this.out_depth, 0.0); // tanh(C) -> to simplify bp computation
    },
    
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
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
        json.filters.push(this.filters[i].toJSON());
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
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
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
  
  
  
  global.LSTMLayer = LSTMLayer;
  global.LSTMRecurrentLayer = LSTMRecurrentLayer;
  
})(convnetjs);

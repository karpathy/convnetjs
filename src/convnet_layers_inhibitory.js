// this file implements input and output gates

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  // Input Gate
  // Transfer function is step (to simulate sigmoid), with an arbitrary threshold
  // output is 0 to 1
  // This layer is similar to activation function layer, and it ONLY WORKS FOR
  //    - "fc"
  // At network constructor, it adds inGateNum (N) to fc's depth, and then it took the last (N) dotproduct output as the input to inGate
  // TODO to support convolution layer (it should work in frequency domain) 
  
  var InputGateLayer = function(opt){
    var opt = opt || {};
    
    //required
    this.gate_depth = typeof opt.gate_depth !== 'undefined' ? opt.gate_depth : 1;
    
    // optional
    this.min_val = typeof opt.min_val !== 'undefined' ? opt.min_val : 0;
    this.max_val = typeof opt.max_val !== 'undefined' ? opt.max_val : 1.0;
    this.threshold = typeof opt.threshold !== 'undefined' ? opt.threshold : 0.3; //arbitrary magic number
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;
    
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth_temp = opt.in_depth;
    
        
    this.out_depth = this.out_depth_temp - this.gate_depth;
    this.num_inputs = this.out_sx * this.out_sy *this.out_depth;
    
    // init filter
    this.filters = [];
    for(var i=0;i<this.num_inputs ;i++) { this.filters.push(new Vol(1, 1, this.gate_depth)); }
    
    this.layer_type = 'ingate';
    this.init();
  }
  InputGateLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      this.gate_act_aggregate = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      this.gate_act = new Vol(this.out_sx, this.out_sy, this.gate_depth, 0.0);
    },
    
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act.setConst(0.0);
      this.gate_act.setConst(0.0);
      this.gate_act_aggregate.setConst(0.0);
      
      //perform step gate transform
      //TODO reduce complexity of for-loops
      for(var x = 0; x < this.out_sx; x++){
        for(var y = 0; y < this.out_sy; y++){
          for(var d = 0; d < this.gate_depth; d++){
            var gateAct = (this.in_act.get(x,y,d + this.out_depth) < this.threshold) ? this.min_val : this.max_val;
            this.gate_act.set(x,y,d, gateAct);
          }
        }
      }
      
      for(var x = 0; x < this.out_sx; x++){
        for(var y = 0; y < this.out_sy; y++){
          for(var d = 0; d < this.out_depth; d++){
            var iOut = ((this.out_sx * y)+x)*this.out_depth+d;
            var iIn = ((this.out_sx * y)+x)*(this.out_depth + this.gate_depth) +d;
            
            var gateSum = 0;
            
            //go through filter
            for(var wd = 0; wd < this.gate_depth; wd++){
              gateSum += this.filters[iOut].w[wd] * this.gate_act.get(x,y,wd);
            }
            
            this.gate_act_aggregate.w[iOut] = gateSum;
            
            console.log(gateSum > this.threshold);
            
            //============ IMPORTANT==============
            // inhibitaion function (large gateSume -> 0 throughput)
            // reusing threshold as it is an arbitrary positive number anyway
            var output = (gateSum > this.threshold)? 0 : this.in_act.w[iIn];
            
            this.out_act.w[iOut] = output;
          }
        }
      }
      
      return this.out_act;
    },
    
    backward: function() {
      this.in_act.setGradConst(0.0);
      this.gate_act.setGradConst(0.0);
      
      for(var x = 0; x < this.out_sx; x++){
        for(var y = 0; y < this.out_sy; y++){
          for(var d = 0; d < this.out_depth; d++){
            
            var iOut = ((this.out_sx * y)+x)*this.out_depth+d;
            var iIn = ((this.out_sx * y)+x)*(this.out_depth + this.gate_depth) +d;
            
            if(this.out_act.dw[iOut] != 0){
              //reproduce the inhibition field
              var Ygate = (this.gate_act_aggregate.w[iOut] > this.threshold)? this.min_val : this.max_val;
              this.in_act.dw[iIn] = this.out_act.dw[iOut] * Ygate;
              
              //propagate through the gates
              var dCdYgate = this.out_act.dw[iOut] * this.in_act.w[iIn];
              var indicator1 = (dCdYgate > 0)? this.min_val : this.max_val;
              
              //=============IMPORTANT================
              // apply NEGATIVE step function derivative here (because it is inhibitory)
              this.gate_act_aggregate.dw[iOut] = -global.stepFunctionBP(Ygate, indicator1);
            }
          }
        }
      }
      
      // BP the inhibitory gates
      // TODO reduce for loops
      for(var x = 0; x < this.out_sx; x++){
        for(var y = 0; y < this.out_sy; y++){
          for(var d = 0; d < this.out_depth; d++){
            iOut = ((this.out_sx * y)+x)*this.out_depth+d;
            for(var wd = 0; wd < this.gate_depth; wd++){
              var dCdYagg = this.gate_act_aggregate.get_grad(x,y,d);
              
              this.filters[iOut].dw[wd] = dCdYagg * this.gate_act.get(x,y,wd);
              this.gate_act.add_grad(x,y,wd, dCdYagg *  this.filters[iOut].w[wd]);
            }
          }
        }
      }
      
      //apply step function bp to gate_act
      for(var x = 0; x < this.out_sx; x++){
        for(var y = 0; y < this.out_sy; y++){
          for(var wd = 0; wd < this.gate_depth; wd++){
            var indicator2 = (this.gate_act.get_grad(x,y,wd) > 0)? this.min_val : this.max_val;
            this.in_act.set_grad(x,y,wd + this.out_depth, global.stepFunctionBP(this.in_act.get(x,y,wd + this.out_depth), indicator2));
          }
        }
      }
    },
    
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.filters.length;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
      }
      return response;
    },
    
    toJSON: function() {
      var json = {};
      json.min_val = this.min_val;
      json.max_val = this.max_val;
      json.threshold = this.threshold;
      
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      
      
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      
      json.gate_depth = this.gate_depth;
      json.out_depth = this.out_depth;
      
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
      
    },
    fromJSON: function(json) {
      this.min_val = json.min_val;
      this.max_val = json.max_val;
      this.threshold = json.threshold || 0.5;
      this.layer_type = json.layer_type;
      
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }

      this.gate_depth = json.gate_depth;
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
      
      //init
      this.init();
    }
  }
  
  global.InputGateLayer = InputGateLayer;

})(convnetjs);


(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  var BufferLayer = function(opt) {
      this.init(opt);
  }
  
  BufferLayer.prototype = {
    init : function(opt){
      var opt = opt || {};

      // required
      this.in_depth = opt.in_depth;
      this.in_sx = opt.in_sx;
      this.in_sy = opt.in_sy;
  
      // optional
      this.bufferSize = typeof opt.bufferSize !== 'undefined' ? opt.bufferSize : 4; 
      
      // computed
      this.out_sx = this.in_sx;
      this.out_sy = this.in_sy;
      this.out_depth = this.in_depth*this.bufferSize;
      this.oneBufferSize = this.out_sx * this.out_sy * this.in_depth;
      
      this.cyclicBufferCnt = 0;
      
      this.layer_type = 'buffer';
      // store switches for x,y coordinates for where the max comes from, for each output neuron
      
      this.bufferStore = global.zeros(this.out_sx*this.out_sy*this.out_depth);
      this.out_act = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
    },

    reset: function(){
      //empty the buffer
      for(var i = 0; i < this.bufferStore.length; i++){
        this.bufferStore[i] = 0;
      }
    },
    
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act.setConst(0.0);
      
      //fill in the new value
      for(var b=0; b<this.bufferSize; b++){
        var startingPoint = ((this.cyclicBufferCnt + this.bufferSize - b) % this.bufferSize) * this.oneBufferSize;
        for(var d=0;d<this.in_depth;d++) {
          for(var x=0; x<this.out_sx; x++) {
            for(var y=0; y<this.out_sy; y++) {
              if(b == 0){
                //inputting current value
                this.bufferStore[startingPoint + x + y + d] = V.get(x,y,d);
              }
              this.out_act.set(x, y, d, this.bufferStore[startingPoint + x + y + d]);
            }
          }
        }
      }
     
      this.cyclicBufferCnt = (this.cyclicBufferCnt+1) % this.bufferSize;
      return this.out_act;
    },
    
    backward: function() { 
      // pooling layers have no parameters, so simply compute 
      // gradient wrt data here
      this.in_act.setGradConst(0.0);
      for(var d=0;d<this.in_depth;d++) {
        for(var x=0; x<this.out_sx; x++) {
          for(var y=0; y<this.out_sy; y++) {
            var value = 0;
            for(var b=0; b<this.bufferSize; b++){
              value += this.out_act.get_grad[x, y, b * this.in_depth + d];
            }
            value = value / this.bufferSize;
            this.in_act.set_grad(x, y, d, value); 
          }
        }
      }
    },
    
    getParamsAndGrads: function() {
      return [];
    },
    
    toJSON: function() {
      var json = {};
      json.bufferSize = this.bufferSize;
      
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
      this.bufferSize = json.bufferSize;
      this.in_depth = json.in_depth;
      
      this.oneBufferSize = this.out_sx * this.out_sy * this.in_depth;
      this.cyclicBufferCnt = 0;
      
      this.bufferStore = global.zeros(this.out_sx*this.out_sy*this.out_depth);
      this.out_act = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
    }
  }

  global.BufferLayer = BufferLayer;
  
})(convnetjs);

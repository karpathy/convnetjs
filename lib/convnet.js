var convnetjs = (function(exports){

  // a 3D volume of float activations (and their gradients)
  // of size sx,sy,depth
  // c is optionally a value to initialize the volume with
  // if c is missing, fills Vol with random numbers drawn from N(0, 0.01)
  var Vol = function(sx, sy, depth, c) {
    this.sx = sx;
    this.sy = sy;
    this.depth = depth;

    /*
    if(typeof c === 'undefined') {
      // initialize w with small random weights if c not provided
      this.w = [];
      this.dw = [];
      var scale = Math.sqrt(1.0/(sx*sy*depth));
      for(var i=0;i<sx*sy*depth;i++) { 
        //this.w.push(randn(0.0, 0.01)); 
        //this.w.push(randf(-scale, scale));
        this.w.push(randn(0.0, scale));
        this.dw.push(0.0); 
      }
    } else {
      this.w = [];
      this.dw = [];
      for(var i=0;i<sx*sy*depth;i++) { 
        this.w.push(c); 
        this.dw.push(0.0); 
      }
    }*/

    //Array buffer implementation
    var length = sx*sy*depth;
    var scale = Math.sqrt(1.0/length);
    this.w = new Float64Array(length*2);
    this.dw = new Float64Array(length*2);
    for(var i=0;i<length;i++) {
      this.w[i] = typeof c === 'undefined' ? randn(0.0, scale) : c;
      this.dw[i] = 0.0;
    }
  }

  Vol.prototype = {
    get: function(x, y, d) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      //if(ix > this.w.length) console.log('GET PROBLEM ' + ix + ' but ' + this.w.length);
      return this.w[ix]; 
    },
    set: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      //if(ix > this.w.length) console.log('SET PROBLEM ' + ix + ' but ' + this.w.length);
      this.w[ix] = v; 
    },
    add: function(x, y, d, v) { this.w[((this.sx * y)+x)*this.depth+d] += v; },

    get_grad: function(x, y, d) { return this.dw[((this.sx * y)+x)*this.depth+d]; },
    set_grad: function(x, y, d, v) { this.dw[((this.sx * y)+x)*this.depth+d] = v; },
    add_grad: function(x, y, d, v) { this.dw[((this.sx * y)+x)*this.depth+d] += v; },

    cloneAndZero: function() { return new Vol(this.sx, this.sy, this.depth, 0.0)},
    clone: function() {
      var V = new Vol(this.sx, this.sy, this.depth, 0.0);
      for(var i=0;i<this.w.length;i++) { V.w[i] = this.w[i]; }
      return V;
    },
    addFrom: function(V) { for(var k=0;k<this.w.length;k++) { this.w[k] += V.w[k]; }},
    addFromScaled: function(V, a) { for(var k=0;k<this.w.length;k++) { this.w[k] += a*V.w[k]; }},
    setConst: function(a) { for(var k=0;k<this.w.length;k++) { this.w[k] = a; }},

    toJSON: function() {
      var json = {}
      json.sx = this.sx; 
      json.sy = this.sy;
      json.depth = this.depth;
      json.w = this.w;
      return json;
      // we wont back up gradients to save space
    },
    fromJSON: function(json) {
      this.sx = json.sx;
      this.sy = json.sy;
      this.depth = json.depth;
      this.w = json.w;
      this.dw = zeros(this.w.length);
    }
  }

  /*
  Below we define layers. Every layer must implement:
  - forward()
  - backward()
  - getParamsAndGrads()
  and have data members:
  - .out_sx, .out_sy, .out_depth
  */

  // construct a convolutional layer
  // currently, we do convolution in 'same' Matlab mode, not 'valid'
  // this might be an option for future though
  var ConvLayer = function(sx, sy, out_depth, stride, in_sx, in_sy, in_depth) {
    this.sx = sx; // filter size in x, y dims
    this.sy = sy;
    this.stride = stride;
    this.in_depth = in_depth; // depth of input volume
    this.out_depth = out_depth; // depth of output volume (also #filters in this layer)
    this.out_sx = Math.floor(in_sx / stride); // compute size of output volume
    this.out_sy = Math.floor(in_sy / stride);
    this.layer_type = 'conv';

    this.filters = [];
    for(var i=0;i<out_depth;i++) { this.filters.push(new Vol(sx, sy, in_depth)); }
    this.biases = new Vol(1, 1, out_depth, 0.1);
    //this.fracs = zeros(out_depth);
  }
  ConvLayer.prototype = {
    forward: function(V) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);

      for(var d=0;d<this.out_depth;d++) {
        
        var f = this.filters[d];
        var ax=0; // convenience pointers to output array x and y
        var ay=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {

            // convolve centered at this particular location
            // could be bit more efficient, going for correctness first
            var a = 0.0;
            for(var fx=0;fx<f.sx;fx++) {
              for(var fy=0;fy<f.sy;fy++) {
                for(var fd=0;fd<f.depth;fd++) {
                  var oy = y+fy-hy;
                  var ox = x+fx-hx;
                  if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                    //a += f.get(fx, fy, d) * V.get(ox, oy, d);
                    // avoid function call overhead for efficiency, compromise modularity :(
                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V.sx * oy)+ox)*V.depth+fd];
                  }
                }
              }
            }
            a += this.biases.w[d];

            // relu threshold and assign
            if(a>0) {
              A.set(ax, ay, d, a);
            }
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() { 

      // compute gradient wrt weights, biases and input data
      var V = this.in_act;
      V.dw = zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var ax=0; // convenience pointers to output array x and y
        var ay=0;

        //var num_locs=0;
        //var num_fired=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {
            //num_locs++;
            if(this.out_act.get(ax,ay,d) === 0.0) { continue; } // no activaiton => no gradient
            //num_fired++;

            // convolve and add up the gradients. 
            // could be more efficient, going for correctness first
            var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            for(var fx=0;fx<f.sx;fx++) {
              for(var fy=0;fy<f.sy;fy++) {
                for(var fd=0;fd<f.depth;fd++) {
                  var oy = y+fy-hy;
                  var ox = x+fx-hx;
                  if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                    // avoid function call overhead and use Vols directly for efficiency
                    f.dw[((f.sx * fy)+fx)*f.depth+fd] += V.w[((V.sx * oy)+ox)*V.depth+fd]*chain_grad;
                    V.dw[((V.sx * oy)+ox)*V.depth+fd] += f.w[((f.sx * fy)+fx)*f.depth+fd]*chain_grad;

                  }
                }
              }
            }
            this.biases.dw[d] += chain_grad;
          }
        }
        //var frac_fired = num_fired/num_locs;
        // experimental: truncated L1 penalty for dealing with dead RELUs?
        //if(frac_fired<0.05) this.biases.dw[d] -= 0.001;
        //if(frac_fired>0.2) this.biases.dw[d] += 0.001;
        //this.fracs[d] = frac_fired;
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, decay: true});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, decay: false});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth; // depth of input volume
      json.out_depth = this.out_depth; // depth of output volume (also #filters in json layer)
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth; // depth of output volume (also #filters in this layer)
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  var PoolLayer = function(sx, sy, stride, in_sx, in_sy, in_depth) {
    this.sx = sx;
    this.sy = sy;
    this.in_depth = in_depth;
    this.out_depth = in_depth;
    this.out_sx = Math.floor(in_sx / stride);
    this.out_sy = Math.floor(in_sy / stride);
    this.stride = stride;
    this.layer_type = 'pool';

    // store switches for x,y coordinates for where the max comes from,
    // for each output neuron
    this.switchx = zeros(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = zeros(this.out_sx*this.out_sy*this.out_depth);
  }

  PoolLayer.prototype = {
    forward: function(V) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);
      
      var n=0; // a counter for switches
      for(var d=0;d<this.out_depth;d++) {
        var ax=0; // convenience pointers to output array x and y
        var ay=0;

        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {

            // convolve centered at this particular location
            var a = -99999; // hopefully small enough ;\
            var winx=-1,winy=-1;
            for(var fx=0;fx<this.sx;fx++) {
              for(var fy=0;fy<this.sy;fy++) {
                var oy = y+fy-hy;
                var ox = x+fx-hx;
                if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                  var v = V.get(ox, oy, d);
                  // perform max pooling and store pointers to where
                  // the max came from. This will speed up backprop 
                  // and can help make nice visualizations in future
                  if(v > a) { a = v; winx=ox; winy=oy;}
                }
              }
            }
            this.switchx[n] = winx;
            this.switchy[n] = winy;
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
      V.dw = zeros(V.w.length); // zero out gradient wrt data
      var A = this.out_act; // computed in forward pass 

      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);
      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var ax=0; // convenience pointers to output array x and y
        var ay=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {
            
            var chain_grad = this.out_act.get_grad(ax,ay,d);
            V.add_grad(this.switchx[n], this.switchy[n],d,chain_grad);
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
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth; // depth of input volume
      json.out_depth = this.out_depth; // depth of output volume (also #filters in json layer)
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth; // depth of output volume (also #filters in this layer)
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
    }
  }

  var FullyConnLayer = function(num_neurons, num_inputs) {
    this.num_inputs = num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.out_depth = num_neurons;
    this.layer_type = 'fc';

    this.filters = [];
    for(var i=0;i<num_neurons ;i++) { this.filters.push(new Vol(1, 1, num_inputs)); }
    this.biases = new Vol(1, 1, num_neurons, 0.1);
  }

  FullyConnLayer.prototype = {
    forward: function(V) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      for(var i=0;i<this.out_depth;i++) {
        var a = 0.0;
        var wi = this.filters[i].w;
        for(var d=0;d<this.num_inputs;d++) {
          a += V.w[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        if(a < 0) a = 0; // relu
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      V.dw = zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(var i=0;i<this.out_depth;i++) {
        //if(this.out_act.get(0,0,i) === 0.0) { continue; } // no activation => no gradient
        if(this.out_act.w[i] === 0.0) { continue; } // no activation => no gradient
        var tfi = this.filters[i];
        var chain_grad = this.out_act.dw[i];

        for(var d=0;d<this.num_inputs;d++) {
          V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
          tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
        }
        this.biases.dw[i] += chain_grad;
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, decay: true});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, decay: false});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth; // depth of output volume (also #filters in json layer)
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth; // depth of output volume (also #filters in this layer)
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  // we are creating mostly to hold pointer to net input
  // and let the net backprop gradient wrt data, if we like, in future.
  var InputLayer = function(out_sx, out_sy, out_depth) {
    this.out_sx = out_sx; 
    this.out_sy = out_sy;
    this.out_depth = out_depth;
    this.layer_type = 'input';
  }
  InputLayer.prototype = {
    forward: function(V) {
      this.in_act = V;
      this.out_act = V;
      return this.out_act; // dummy identity function for now
    },
    backward: function() { return; },
    getParamsAndGrads: function() {
      return [];
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth; // depth of output volume (also #filters in json layer)
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth; // depth of output volume (also #filters in json layer)
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
    }
  }

  // experimental layer for now. I think it works but I'm not 100%
  // the gradient check is a bit funky. I'll look into this a bit later.
  // Local Response Normalization in window, along depths of volumes
  var LocalResponseNormalizationLayer = function(k, n, alpha, beta, out_sx, out_sy, out_depth) {
    this.k = k;
    this.n = n;
    this.alpha = alpha; // normalize by size
    this.beta = beta;
    this.out_sx = out_sx; 
    this.out_sy = out_sy;
    this.out_depth = out_depth;
    this.layer_type = 'lrn';
    if(n%2 === 0) {console.log('WARNING n should be odd for LRN layer');}
  }
  LocalResponseNormalizationLayer.prototype = {
    forward: function(V) {
      this.in_act = V;

      var A = V.cloneAndZero();
      this.S_cache_ = V.cloneAndZero();
      var n2 = Math.floor(this.n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var ai = V.get(x,y,i);

            // normalize in a window of size n
            var den = 0.0;
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {
              var aa = V.get(x,y,j);
              den += aa*aa;
            }
            den *= this.alpha / this.n;
            den += this.k;
            this.S_cache_.set(x,y,i,den); // will be useful for backprop
            den = Math.pow(den, this.beta);
            A.set(x,y,i,ai/den);
          }
        }
      }

      this.out_act = A;
      return this.out_act; // dummy identity function for now
    },
    backward: function() { 
      // evaluate gradient wrt data
      var V = this.in_act; // we need to set dw of this
      V.dw = zeros(V.w.length); // zero out gradient wrt data
      var A = this.out_act; // computed in forward pass 

      var n2 = Math.floor(this.n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var chain_grad = this.out_act.get_grad(x,y,i);
            var S = this.S_cache_.get(x,y,i);
            var SB = Math.pow(S, this.beta);
            var SB2 = SB*SB;

            // normalize in a window of size n
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {              
              var aj = V.get(x,y,j); 
              var g = -aj*this.beta*Math.pow(S,this.beta-1)*this.alpha/this.n*2*aj;
              if(j===i) g+= SB;
              g /= SB2;
              g *= chain_grad;
              V.add_grad(x,y,j,g);
            }

          }
        }
      }
    },
    getParamsAndGrads: function() { return []; },
    toJSON: function() {
      var json = {};
      json.k = this.k;
      json.n = this.n;
      json.alpha = this.alpha; // normalize by size
      json.beta = this.beta;
      json.out_sx = this.out_sx; 
      json.out_sy = this.out_sy;
      json.out_depth = this.out_depth;
      json.layer_type = this.layer_type;
      return json;
    },
    fromJSON: function(json) {
      this.k = json.k;
      this.n = json.n;
      this.alpha = json.alpha; // normalize by size
      this.beta = json.beta;
      this.out_sx = json.out_sx; 
      this.out_sy = json.out_sy;
      this.out_depth = json.out_depth;
      this.layer_type = json.layer_type;
    }
  }

  // labels are assumed to be ints 0..N for now
  var SoftmaxLayer = function(num_classes, num_inputs) {
    this.num_inputs = num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.out_depth = num_classes;
    this.layer_type = 'softmax';

    this.filters = [];
    for(var i=0;i<num_classes ;i++) { 
      this.filters.push(new Vol(1, 1, num_inputs)); 
    }
    this.biases = new Vol(1, 1, num_classes, 0.0);
  }

  SoftmaxLayer.prototype = {
    forward: function(V) {
      this.in_act = V;

      var A = new Vol(1, 1, this.out_depth, 0.0);
      var as = [];
      var amax;
      for(var i=0;i<this.out_depth;i++) {
        var a = 0.0;
        var wi = this.filters[i].w;
        for(var d=0;d<this.num_inputs;d++) {
          a += V.w[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        as.push(a);
        if(i===0 || a > amax) { amax = a; } // keep track of max element
      }

      // compute exponentials (carefully to not blow up)
      var es = [];
      var esum = 0.0;
      for(var i=0;i<this.out_depth;i++) {
        var e = Math.exp(as[i] - amax);
        esum += e;
        es.push(e);
      }

      // normalize and output to sum to one
      for(var i=0;i<this.out_depth;i++) {
        es[i] = es[i] / esum;
        A.set(0, 0, i, es[i]);
      }

      this.es = es; // save these for backprop
      this.out_act = A;
      return this.out_act;
    },
    backward: function(y) {

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = zeros(x.w.length); // zero out the gradient of input Vol

      for(var i=0;i<this.out_depth;i++) {
        var indicator = 0.0;
        if(i===y) indicator = 1.0;
        var mul = -(indicator - this.es[i]);
        var tfi = this.filters[i];
        for(var d=0;d<this.num_inputs;d++) {
          tfi.dw[d] += x.w[d] * mul;
        }
        this.biases.dw[i] += mul;

        // and wrt data
        for(var d=0;d<this.num_inputs;d++) {
          x.dw[d] += tfi.w[d] * mul;
        }
      }

      // loss is the class negative log likelihood
      return -Math.log(this.es[y]);
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, decay: true});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, decay:false});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth; // depth of output volume (also #filters in json layer)
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth; // depth of output volume (also #filters in this layer)
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer softmax
  var Net = function(options) {
    this.layers = [];
  }

  Net.prototype = {
    
    // takes a list of layer definitions and creates the network layer objects
    makeLayers: function(defs) {

      // few checks for now
      if(defs.length<2) {console.log('ERROR! For now at least have input and softmax layers.');}
      if(defs[0].type !== 'input') {console.log('ERROR! For now first layer should be input.');}
      if(defs[defs.length-1].type !== 'softmax') {console.log('ERROR! For now last layer should be softmax.');}

      this.layers = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        var type = def.type;
        var prev;
        if(i > 0) { prev = this.layers[i-1]; }

        if(type === 'fc') {

          // fully connected layer
          var num_inputs = prev.out_sx * prev.out_sy * prev.out_depth;
          var f = new FullyConnLayer(def.filters, num_inputs);
          this.layers.push(f);
        } else if(type === 'lrn') {

          var r = new LocalResponseNormalizationLayer(def.k, def.n, def.alpha, def.beta, prev.out_sx, prev.out_sy, prev.out_depth);
          this.layers.push(r);

        } else if(type === 'input') {

          // input layer, declares input size
          var il = new InputLayer(def.out_sx, def.out_sy, def.out_depth);
          this.layers.push(il);

        } else if(type === 'softmax') {

          var num_inputs = prev.out_sx * prev.out_sy * prev.out_depth;
          var s = new SoftmaxLayer(def.num_classes, num_inputs);
          this.layers.push(s);

        } else if(type === 'conv' || type === 'pool') {

          // some common options and defaults
          var sx = def.sx;
          var sy = def.sy || sx; // assume square filters

          // layer type specific options and defaults
          if(type === 'conv') {
            var out_depth = def.filters;
            var stride = def.stride || 1;
            var c = new ConvLayer(sx, sy, out_depth, stride, prev.out_sx, prev.out_sy, prev.out_depth);
            this.layers.push(c);
          }
          else if(type === 'pool') {
            var stride = def.stride || 2;
            var p = new PoolLayer(sx, sy, stride, prev.out_sx, prev.out_sy, prev.out_depth);
            this.layers.push(p);
          }
        } else {
          console.log('UNRECOGNIZED LAYER TYPE');
        }
      }
    },

    // forward prop the network
    forward: function(V) {
      var act = this.layers[0].forward(V);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act);
      }
      return act;
    },

    // backprop: compute gradients wrt all parameters
    backward: function(y) {
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y); // last layer assumed softmax
      for(var i=N-2;i>=0;i--) { // first layer assumed input
        this.layers[i].backward();
      }
      return loss;
    },
    getParamsAndGrads: function() {
      // accumulate parameters and gradients for the entire network
      var response = [];
      for(var i=0;i<this.layers.length;i++) {
        var layer_reponse = this.layers[i].getParamsAndGrads();
        for(var j=0;j<layer_reponse.length;j++) {
          response.push(layer_reponse[j]);
        }
      }
      return response;
    },
    getPrediction: function() {
      var S = this.layers[this.layers.length-1]; // softmax layer
      var p = S.out_act.w;
      var maxv = p[0];
      var maxi = 0;
      for(var i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i;}
      }
      return maxi;
    },
    toJSON: function() {
      var json = {};
      json.layers = [];
      for(var i=0;i<this.layers.length;i++) {
        json.layers.push(this.layers[i].toJSON());
      }
      return json;
    },
    fromJSON: function(json) {
      this.layers = [];
      for(var i=0;i<json.layers.length;i++) {
        var Lj = json.layers[i]
        var t = Lj.layer_type;
        var L;
        if(t==='input') { L = new InputLayer(); }
        if(t==='conv') { L = new ConvLayer(); }
        if(t==='pool') { L = new PoolLayer(); }
        if(t==='lrn') { L = new LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new SoftmaxLayer(); }
        if(t==='fc') { L = new FullyConnLayer(); }
        L.fromJSON(Lj);
        this.layers.push(L);
      }
      return json;
    }
  }
  
  /*
  Learning rate: must be set carefully by trial and error and annealed 
    over time several orders of magnitude. Usually numbers around 0.1 work.
    This is a very important parameter to tune while network trains.
  Decay: the amount of weight decay. Increase it to penalize high weights
    (this regularizes the network and is a good idea if you're seeing overfitting,
    where your training accuracy is much higher than validation accuracy)
  Batch Size: usually set to around 16,32,128 or 256. Generally higher is 
    better because you're getting more accurate gradients but this significantly
    slows down the training.
  Momentum: exponentially averages gradients over time to smooth the network's
    progress. Directions with high gradient variance will slow down and directions
    with consistent gradient will speed up progress along. 0.9 is a reasonable
    setting and you almos always want this to be between 0.6-0.95 as it really
    helps with convergence times.  
  */ 
  var SGDTrainer = function(net, options) {

    this.net = net;

    var options = options || {};
    this.learning_rate = options.learning_rate || 0.01;
    this.decay = options.decay || 0.001;
    this.batch_size = options.batch_size || 1;

    this.momenum = 0.9;
    if(typeof options.momentum !== 'undefined') this.momentum = options.momentum;
    this.k = 0; // iteration counter

    this.last_gs = []; // last iteration gradients (used for momentum calculations)
  }

  SGDTrainer.prototype = {
    train: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x); 
      var end = new Date().getTime();
      var fwd_time = end - start;

      var start = new Date().getTime();
      var softmax_loss = this.net.backward(y);
      var decay_loss = 0.0;
      var end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        // initialize lists for momentum keeping. Will only run first iteration
        var pglist = this.net.getParamsAndGrads();
        if(this.last_gs.length === 0 && this.momentum > 0.0) {
          for(var i=0;i<pglist.length;i++) {
            this.last_gs.push(zeros(pglist[i].params.length));
          }
        }

        // perform an update for all sets of weights
        for(var i=0;i<pglist.length;i++) {
          var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
          var p = pg.params;
          var g = pg.grads;

          // learning rate for some parameters.
          var decay = 0.0;
          if(pg.decay) { decay = this.decay; }

          for(var j=0;j<p.length;j++) {
            decay_loss += decay*p[j]*p[j]/2; // accumulate weight decay loss
            if(this.momentum > 0.0) {
              // back up the last gradients and do weighted update
              var dir = -this.learning_rate * (decay * p[j] +  g[j]) / this.batch_size;
              var dir_adj = this.momentum * this.last_gs[i][j] + (1.0 - this.momentum) * dir;
              p[j] += dir_adj;
              this.last_gs[i][j] = dir_adj;
            } else {
              // vanilla sgd
              p[j] -= this.learning_rate * (decay * p[j] +  g[j]) / this.batch_size;
            }
            g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
          }
        }
      }

      return {fwd_time: fwd_time, bwd_time: bwd_time, softmax_loss: softmax_loss, decay_loss: decay_loss, loss: softmax_loss + decay_loss}
    }
  }

  // Misc utility functions
  // for random numbers
  function randf(a, b) { return Math.random()*(b-a)+a; }
  function randi(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  function randn(mu, std){ return mu+gaussRandom()*std; }
  function gaussRandom() {
      var u = 2*Math.random()-1;
      var v = 2*Math.random()-1;
      var r = u*u + v*v;
      if(r == 0 || r > 1) return gaussRandom();
      var c = Math.sqrt(-2*Math.log(r)/r);
      return u*c; // also v*c but ah well
  }

  // array stuff
  function zeros(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    return new Float64Array(n*2);
    //for(var i=0;i<n;i++) { arr[i]= 0; }
    //return arr;
  }

  // volume stuff
  function augment(V, crop, fliplr) {
    // note assumes square images for now
    if(typeof(fliplr)==='undefined') var fliplr = false;

    // randomly sample a crop in the input volume
    var W;
    if(crop < V.sx) {
      W = new Vol(crop, crop, V.depth, 0.0);
      var dx = randi(0, V.sx - crop); // sample a cropping offset
      var dy = randi(0, V.sy - crop);
      for(var x=0;x<crop;x++) {
        for(var y=0;y<crop;y++) {
          for(var d=0;d<V.depth;d++) {
           W.set(x,y,d,V.get(x+dx,y+dy,d)); // copy data over
          }
        }
      }
    } else {
      W = V;
    }

    if(fliplr) {
      // flip volume horziontally
      var W2 = W.cloneAndZero();
      for(var x=0;x<W.sx;x++) {
        for(var y=0;y<W.sy;y++) {
          for(var d=0;d<W.depth;d++) {
           W2.set(x,y,d,W.get(W.sx - x - 1,y,d)); // copy data over
          }
        }
      }
      W = W2; //swap
    }
    return W;
  }

  // img is the DOM element that contains a loaded image
  // returns a Vol of size (W, H, 4). 4 is for RGBA
  function img_to_vol(img, convert_grayscale) {

    if(typeof(convert_grayscale)==='undefined') var convert_grayscale = false;

    var canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext("2d");

    // due to a Firefox bug
    try {
      ctx.drawImage(img, 0, 0);
    } catch (e) {
      if (e.name === "NS_ERROR_NOT_AVAILABLE") {
        // sometimes happens, lets just abort
        return false;
      } else {
        throw e;
      }
    }

    try {
      var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    } catch (e) {
      if(e.name === 'IndexSizeError') {
        return false; // not sure what causes this sometimes but okay abort
      } else {
        throw e;
      }
    }

    // prepare the input: get pixels and normalize them
    var p = img_data.data;
    var W = img.width;
    var H = img.height;
    var pv = []
    for(var i=0;i<p.length;i++) {
      pv.push(p[i]/255.0-0.5); // normalize
    }
    var x = new Vol(W, H, 4, 0.0); //input volume (image)
    x.w = pv;

    if(convert_grayscale) {
      // flatten into depth=1 array
      var x1 = new Vol(W, H, 1, 0.0);
      for(var i=0;i<W;i++) {
        for(var j=0;j<H;j++) {
          x1.set(i,j,0,x.get(i,j,0));
        }
      }
      x = x1;
    }

    return x;
  }

  // export public members
  exports = exports || {};
  exports.Net = Net;
  exports.ConvLayer = ConvLayer;
  exports.PoolLayer = PoolLayer;
  exports.SoftmaxLayer = SoftmaxLayer;
  exports.LocalResponseNormalizationLayer = LocalResponseNormalizationLayer;
  exports.Vol = Vol;
  exports.SGDTrainer = SGDTrainer;
  exports.augment = augment;
  exports.zeros = zeros;
  exports.img_to_vol = img_to_vol;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js



var convnetjs = convnetjs || { REVISION: 'ALPHA' };
(function(global) {
  "use strict";

  // Random number utilities
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  var randf = function(a, b) { return Math.random()*(b-a)+a; }
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  // Array utilities
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  // return max and min of a given non-empty array.
  var maxmin = function(w) {
    if(w.length === 0) { return {}; } // ... ;s
    var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    var n = w.length;
    for(var i=1;i<n;i++) {
      if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
      if(w[i] < minv) { minv = w[i]; mini = i; } 
    }
    return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
  }

  global.randf = randf;
  global.randi = randi;
  global.randn = randn;
  global.zeros = zeros;
  global.maxmin = maxmin;
  
})(convnetjs);
(function(global) {
  "use strict";

  // Vol is the basic building block of all data in a net.
  // it is essentially just a 3D volume of numbers, with a
  // width (sx), height (sy), and depth (depth).
  // it is used to hold data for all filters, all volumes,
  // all weights, and also stores all gradients w.r.t. 
  // the data. c is optionally a value to initialize the volume
  // with. If c is missing, fills the Vol with random numbers.
  var Vol = function(sx, sy, depth, c) {
    this.sx = sx;
    this.sy = sy;
    this.depth = depth;
    var n = sx*sy*depth;
    this.w = global.zeros(n);
    this.dw = global.zeros(n);
    if(typeof c === 'undefined') {
      // weight normalization is done to equalize the output
      // variance of every neuron, otherwise neurons with a lot
      // of incoming connections have outputs of larger variance
      var scale = Math.sqrt(1.0/(sx*sy*depth));
      for(var i=0;i<n;i++) { 
        this.w[i] = global.randn(0.0, scale);
      }
    } else {
      for(var i=0;i<n;i++) { 
        this.w[i] = c;
      }
    }
  }

  Vol.prototype = {
    get: function(x, y, d) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      return this.w[ix];
    },
    set: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] = v; 
    },
    add: function(x, y, d, v) { this.w[((this.sx * y)+x)*this.depth+d] += v; },

    get_grad: function(x, y, d) { return this.dw[((this.sx * y)+x)*this.depth+d]; },
    set_grad: function(x, y, d, v) { this.dw[((this.sx * y)+x)*this.depth+d] = v; },
    add_grad: function(x, y, d, v) { this.dw[((this.sx * y)+x)*this.depth+d] += v; },

    cloneAndZero: function() { return new Vol(this.sx, this.sy, this.depth, 0.0)},
    clone: function() {
      var V = new Vol(this.sx, this.sy, this.depth, 0.0);
      var n = this.w.length;
      for(var i=0;i<n;i++) { V.w[i] = this.w[i]; }
      return V;
    },
    addFrom: function(V) { for(var k=0;k<this.w.length;k++) { this.w[k] += V.w[k]; }},
    addFromScaled: function(V, a) { for(var k=0;k<this.w.length;k++) { this.w[k] += a*V.w[k]; }},
    setConst: function(a) { for(var k=0;k<this.w.length;k++) { this.w[k] = a; }},

    toJSON: function() {
      // todo: we may want to only save d most significant digits to save space
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
      this.dw = global.zeros(this.w.length);
    }
  }

  global.Vol = Vol;
})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // Volume utilities
  // intended for use with data augmentation
  // crop is the size of output
  // dx,dy are offset wrt incoming volume, of the shift
  // fliplr is boolean on whether we also want to flip left<->right
  var augment = function(V, crop, dx, dy, fliplr) {
    // note assumes square outputs of size crop x crop
    if(typeof(fliplr)==='undefined') var fliplr = false;
    if(typeof(dx)==='undefined') var dx = global.randi(0, V.sx - crop);
    if(typeof(dy)==='undefined') var dy = global.randi(0, V.sy - crop);
    
    // randomly sample a crop in the input volume
    var W;
    if(crop !== V.sx || dx!==0 || dy!==0) {
      W = new Vol(crop, crop, V.depth, 0.0);
      for(var x=0;x<crop;x++) {
        for(var y=0;y<crop;y++) {
          if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) continue; // oob
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

  // img is a DOM element that contains a loaded image
  // returns a Vol of size (W, H, 4). 4 is for RGBA
  var img_to_vol = function(img, convert_grayscale) {

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
      pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
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
  
  global.augment = augment;
  global.img_to_vol = img_to_vol;

})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes: 
  // - FullyConn is fully connected dot products 
  // - ConvLayer does convolutions (so weight sharing spatially)
  // - LocallyConn has local connectivity like ConvLayer, but no weight sharing.
  // putting them together because they are very very similar

  // construct a convolutional layer
  // currently, we do convolution in 'same' Matlab mode, not 'valid'
  // and pad with zeros as necessary. (Though this might be an option in future)
  // The volume is transformed as W1xH1xD1 -> W1/stride x H1/stride x D2
  var ConvLayer = function(opt) {
    var opt = opt || {};

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.sy !== 'undefined' ? opt.stride : 1;
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    this.out_sx = Math.floor(this.in_sx / this.stride); // compute size of output volume
    this.out_sy = Math.floor(this.in_sy / this.stride);
    this.layer_type = 'conv';

    // initializations
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, this.out_depth, 0.1);
  }
  ConvLayer.prototype = {
    forward: function(V, is_training) {
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
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() { 

      // compute gradient wrt weights, biases and input data
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var ax=0; // convenience pointers to output array x and y
        var ay=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {
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
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth;
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
      this.filters = [];
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  var LocallyConnLayer = function(opt) {
    var opt = opt || {};

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.sy !== 'undefined' ? opt.stride : 1;
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    this.out_sx = Math.floor(this.in_sx / this.stride); // compute size of output volume
    this.out_sy = Math.floor(this.in_sy / this.stride);
    this.layer_type = 'local';

    // initializations
    this.filters = [];
    var num_filters = this.out_sx * this.out_sy * this.out_depth;
    for(var i=0;i<num_filters;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, num_filters, 0.1);
  }
  LocallyConnLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);

      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var ax=0; // convenience pointers to output array x and y
        var ay=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {

            var f = this.filters[n];
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
            a += this.biases.w[n];
            A.set(ax, ay, d, a);
            n++;
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() { 

      // compute gradient wrt weights, biases and input data
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
      var hx = Math.floor(this.sx/2.0);
      var hy = Math.floor(this.sy/2.0);
      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var ax=0; // convenience pointers to output array x and y
        var ay=0;
        for(var x=0;x<V.sx;x+=this.stride,ax++) {
          ay = 0;
          for(var y=0;y<V.sy;y+=this.stride,ay++) {
            var f = this.filters[n];
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
            this.biases.dw[n] += chain_grad;
            n++;
          }
        }
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.filters.length;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth;
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
      this.filters = [];
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  var FullyConnLayer = function(opt) {
    var opt = opt || {};

    // required
    // ok fine we will allow 'filters' as the word as well
    this.out_depth = typeof opt.num_neurons !== 'undefined' ? opt.num_neurons : opt.filters;

    // optional 
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'fc';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.1;
    this.filters = [];
    for(var i=0;i<this.out_depth ;i++) { this.filters.push(new Vol(1, 1, this.num_inputs)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }

  FullyConnLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      var Vw = V.w;
      for(var i=0;i<this.out_depth;i++) {
        var a = 0.0;
        var wi = this.filters[i].w;
        for(var d=0;d<this.num_inputs;d++) {
          a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(var i=0;i<this.out_depth;i++) {
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
    }
  }

  global.ConvLayer = ConvLayer;
  global.LocallyConnLayer = LocallyConnLayer;
  global.FullyConnLayer = FullyConnLayer;
  
})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  var PoolLayer = function(opt) {

    var opt = opt || {};

    // required
    this.sx = opt.sx; // filter size
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 2;

    // computed
    this.out_depth = this.in_depth;
    this.out_sx = Math.floor(this.in_sx / this.stride); // compute size of output volume
    this.out_sy = Math.floor(this.in_sy / this.stride);
    this.layer_type = 'pool';
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
  }

  PoolLayer.prototype = {
    forward: function(V, is_training) {
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
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
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
      json.sx = this.sx;
      json.sy = this.sy;
      json.stride = this.stride;
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
      this.sx = json.sx;
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth;
    }
  }

  global.PoolLayer = PoolLayer;

})(convnetjs);

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  var InputLayer = function(opt) {
    var opt = opt || {};

    // this is a bit silly but lets allow people to specify either ins or outs
    this.out_sx = typeof opt.out_sx !== 'undefined' ? opt.out_sx : opt.in_sx;
    this.out_sy = typeof opt.out_sy !== 'undefined' ? opt.out_sy : opt.in_sy;
    this.out_depth = typeof opt.out_depth !== 'undefined' ? opt.out_depth : opt.in_depth;
    this.layer_type = 'input';
  }
  InputLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act = V;
      return this.out_act; // dummy identity function for now
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
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
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
  }

  SoftmaxLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(1, 1, this.out_depth, 0.0);

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
        A.w[i] = es[i];
      }

      this.es = es; // save these for backprop
      this.out_act = A;
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
  }

  RegressionLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act = V;
      return V; // identity function
    },
    // y is a list here of size num_inputs
    backward: function(y) { 

      // compute and accumulate gradient wrt weights and bias of this layer
      var x = this.in_act;
      x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
      var loss = 0.0;
      for(var i=0;i<this.out_depth;i++) {
        var dy = x.w[i] - y[i];
        x.dw[i] = dy;
        loss += 2*dy*dy;
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
    }
  }
  
  global.RegressionLayer = RegressionLayer;
  global.SoftmaxLayer = SoftmaxLayer;

})(convnetjs);

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  // Implements ReLU nonlinearity elementwise
  // x -> max(0, x)
  // the output is in [0, inf)
  var ReluLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'relu';
  }
  ReluLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var V2 = V.clone();
      var N = V.w.length;
      var V2w = V2.w;
      for(var i=0;i<N;i++) { 
        if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
      }
      this.out_act = V2;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act; // we need to set dw of this
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
        else V.dw[i] = V2.dw[i];
      }
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
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
    }
  }

  // Implements Sigmoid nnonlinearity elementwise
  // x -> 1/(1+e^(-x))
  // so the output is between 0 and 1.
  var SigmoidLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'sigmoid';
  }
  SigmoidLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var V2 = V.cloneAndZero();
      var N = V.w.length;
      var V2w = V2.w;
      var Vw = V.w;
      for(var i=0;i<N;i++) { 
        V2w[i] = 1.0/(1.0+Math.exp(-Vw[i]));
      }
      this.out_act = V2;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act; // we need to set dw of this
      var V2 = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        var v2wi = V2.w[i];
        V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i];
      }
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
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
    }
  }
  
  global.ReluLayer = ReluLayer;
  global.SigmoidLayer = SigmoidLayer;

})(convnetjs);

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // An inefficient dropout layer
  // Note this is not most efficient implementation since the layer before
  // computed all these activations and now we're just going to drop them :(
  // same goes for backward pass. Also, if we wanted to be efficient at test time
  // we could equivalently be clever and upscale during train and copy pointers during test
  // todo: make more efficient.
  var DropoutLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'dropout';
    this.drop_prob = typeof opt.drop_prob !== 'undefined' ? opt.drop_prob : 0.5;
    this.dropped = global.zeros(this.out_sx*this.out_sy*this.out_depth);
  }
  DropoutLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      if(typeof(is_training)==='undefined') { is_training = false; } // default is prediction mode
      var V2 = V.clone();
      var N = V.w.length;
      if(is_training) {
        // do dropout
        for(var i=0;i<N;i++) {
          if(Math.random()<this.drop_prob) { V2.w[i]=0; this.dropped[i] = true; } // drop!
          else {this.dropped[i] = false;}
        }
      } else {
        // scale the activations during prediction
        for(var i=0;i<N;i++) { V2.w[i]*=this.drop_prob; }
      }
      this.out_act = V2;
      return this.out_act; // dummy identity function for now
    },
    backward: function() {
      var V = this.in_act; // we need to set dw of this
      var chain_grad = this.out_act;
      var N = V.w.length;
      V.dw = global.zeros(N); // zero out gradient wrt data
      for(var i=0;i<N;i++) {
        if(!(this.dropped[i])) { 
          V.dw[i] = chain_grad.dw[i]; // copy over the gradient
        }
      }
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
      json.drop_prob = this.drop_prob;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
      this.drop_prob = json.drop_prob;
    }
  }
  

  global.DropoutLayer = DropoutLayer;
})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  // a bit experimental layer for now. I think it works but I'm not 100%
  // the gradient check is a bit funky. I'll look into this a bit later.
  // Local Response Normalization in window, along depths of volumes
  var LocalResponseNormalizationLayer = function(opt) {
    var opt = opt || {};

    // required
    this.k = opt.k;
    this.n = opt.n;
    this.alpha = opt.alpha;
    this.beta = opt.beta;

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'lrn';

    // checks
    if(this.n%2 === 0) { console.log('WARNING n should be odd for LRN layer'); }
  }
  LocalResponseNormalizationLayer.prototype = {
    forward: function(V, is_training) {
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
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
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
  

  global.LocalResponseNormalizationLayer = LocalResponseNormalizationLayer;
})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  var Net = function(options) {
    this.layers = [];
  }

  Net.prototype = {
    
    // takes a list of layer definitions and creates the network layer objects
    makeLayers: function(defs) {

      // few checks for now
      if(defs.length<2) {console.log('ERROR! For now at least have input and softmax layers.');}
      if(defs[0].type !== 'input') {console.log('ERROR! For now first layer should be input.');}

      // desugar syntactic for adding activations and dropouts
      var desugar = function() {
        var new_defs = [];
        for(var i=0;i<defs.length;i++) {
          var def = defs[i];
          
          if(def.type==='softmax') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_classes});
          }

          if(def.type==='regression') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_neurons});
          }

          if((def.type==='fc' || def.type==='conv' || def.type==='local') 
              && typeof(def.bias_pref) === 'undefined'){
            def.bias_pref = 0.0;
            if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
              def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
            else { console.log('ERROR unsupported activation ' + def.activation); }
          }
          if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
            new_defs.push({type:'dropout', drop_prob: def.drop_prob});
          }

        }
        return new_defs;
      }
      defs = desugar(defs);

      // create the layers
      this.layers = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        if(i>0) {
          var prev = this.layers[i-1];
          def.in_sx = prev.out_sx;
          def.in_sy = prev.out_sy;
          def.in_depth = prev.out_depth;
        }

        switch(def.type) {
          case 'fc': this.layers.push(new global.FullyConnLayer(def)); break;
          case 'lrn': this.layers.push(new global.LocalResponseNormalizationLayer(def)); break;
          case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
          case 'input': this.layers.push(new global.InputLayer(def)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
          case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
          case 'conv': this.layers.push(new global.ConvLayer(def)); break;
          case 'local': this.layers.push(new global.LocallyConnLayer(def)); break;
          case 'pool': this.layers.push(new global.PoolLayer(def)); break;
          case 'relu': this.layers.push(new global.ReluLayer(def)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
          default: console.log('ERROR: UNRECOGNIZED LAYER TYPE!');
        }
      }
    },

    // forward prop the network
    forward: function(V, is_training) {
      if(typeof(is_training)==='undefined') is_training = false;
      var act = this.layers[0].forward(V, is_training);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
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
        if(t==='input') { L = new global.InputLayer(); }
        if(t==='relu') { L = new global.ReluLayer(); }
        if(t==='sigmoid') { L = new global.SigmoidLayer(); }
        if(t==='dropout') { L = new global.DropoutLayer(); }
        if(t==='conv') { L = new global.ConvLayer(); }
        if(t==='local') { L = new global.LocallyConnLayer(); }
        if(t==='pool') { L = new global.PoolLayer(); }
        if(t==='lrn') { L = new global.LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new global.SoftmaxLayer(); }
        if(t==='regression') { L = new global.RegressionLayer(); }
        if(t==='fc') { L = new global.FullyConnLayer(); }
        L.fromJSON(Lj);
        this.layers.push(L);
      }
    }
  }
  

  global.Net = Net;
})(convnetjs);
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  var SGDTrainer = function(net, options) {

    this.net = net;

    var options = options || {};
    this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
    this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
    this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
    this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
    this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;

    if(typeof options.momentum !== 'undefined') this.momentum = options.momentum;
    this.k = 0; // iteration counter

    this.last_gs = []; // last iteration gradients (used for momentum calculations)
  }

  SGDTrainer.prototype = {
    train: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      var start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      var end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        // initialize lists for momentum keeping. Will only run first iteration
        var pglist = this.net.getParamsAndGrads();
        if(this.last_gs.length === 0 && this.momentum > 0.0) {
          for(var i=0;i<pglist.length;i++) {
            this.last_gs.push(global.zeros(pglist[i].params.length));
          }
        }

        // perform an update for all sets of weights
        for(var i=0;i<pglist.length;i++) {
          var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
          var p = pg.params;
          var g = pg.grads;

          // learning rate for some parameters.
          var l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
          var l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
          var l2_decay = this.l2_decay * l2_decay_mul;
          var l1_decay = this.l1_decay * l1_decay_mul;

          var plen = p.length;
          for(var j=0;j<plen;j++) {
            l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
            l1_decay_loss += l1_decay*Math.abs(p[j]);
            var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
            var l2grad = l2_decay * (p[j]);
            if(this.momentum > 0.0) {
              // back up the last gradients and do weighted update
              var dir = -this.learning_rate * (l2grad + l1grad + g[j]) / this.batch_size;
              var dir_adj = this.momentum * this.last_gs[i][j] + (1.0 - this.momentum) * dir;
              p[j] += dir_adj;
              this.last_gs[i][j] = dir_adj;
            } else {
              // vanilla sgd
              p[j] -= this.learning_rate * (l2grad + l1grad + g[j]) / this.batch_size;
            }
            g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
          }
        }
      }

      // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
      // in future, TODO: have to completely redo the way loss is done around the network as currently 
      // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
      // and it should all be computed correctly and automatically. 
      return {fwd_time: fwd_time, bwd_time: bwd_time, 
              l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
              cost_loss: cost_loss, softmax_loss: cost_loss, 
              loss: cost_loss + l1_decay_loss + l2_decay_loss}
    }
  }
  
  global.SGDTrainer = SGDTrainer;
})(convnetjs);

(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(convnetjs);

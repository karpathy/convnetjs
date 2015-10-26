(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
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

  var arrContains = function(arr, elt) {
    for(var i=0,n=arr.length;i<n;i++) {
      if(arr[i]===elt) return true;
    }
    return false;
  }

  var arrUnique = function(arr) {
    var b = [];
    for(var i=0,n=arr.length;i<n;i++) {
      if(!arrContains(b, arr[i])) {
        b.push(arr[i]);
      }
    }
    return b;
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

  // create random permutation of numbers, in range [0...n-1]
  var randperm = function(n) {
    var i = n,
        j = 0,
        temp;
    var array = [];
    for(var q=0;q<n;q++)array[q]=q;
    while (i--) {
        j = Math.floor(Math.random() * (i+1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
  }

  // sample from list lst according to probabilities in list probs
  // the two lists are of same size, and probs adds up to 1
  var weightedSample = function(lst, probs) {
    var p = randf(0, 1.0);
    var cumprob = 0.0;
    for(var k=0,n=lst.length;k<n;k++) {
      cumprob += probs[k];
      if(p < cumprob) { return lst[k]; }
    }
  }

  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if(typeof field_name === 'string') {
      // case of single string
      return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
    } else {
      // assume we are given a list of string instead
      var ret = default_value;
      for(var i=0;i<field_name.length;i++) {
        var f = field_name[i];
        if (typeof opt[f] !== 'undefined') {
          ret = opt[f]; // overwrite return value
        }
      }
      return ret;
    }
  }

  function assert(condition, message) {
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

    // lstm util function
  var tanh = function(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  
  var sigmoid = function(x){
    return 1.0/(1.0+Math.exp(-x));
  }
  
  // capping value
  var capValue = function(val, max, min){
      var valCap = (val > max)? max : val;
      valCap = (valCap < min)? min : valCap;
      return valCap;
  }
  
  global.randf = randf;
  global.randi = randi;
  global.randn = randn;
  global.zeros = zeros;
  
  global.maxmin = maxmin;
  global.randperm = randperm;
  global.weightedSample = weightedSample;
  
  global.tanh = tanh;
  global.sigmoid = sigmoid;
  global.capValue = capValue;
  
  global.arrUnique = arrUnique;
  global.arrContains = arrContains;
  global.getopt = getopt;
  global.assert = assert;
  
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
    // this is how you check if a variable is an array. Oh, Javascript :)
    if(Object.prototype.toString.call(sx) === '[object Array]') {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.sy = 1;
      this.depth = sx.length;
      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.w = global.zeros(this.depth);
      this.dw = global.zeros(this.depth);
      for(var i=0;i<this.depth;i++) {
        this.w[i] = sx[i];
      }
    } else {
      // we were given dimensions of the vol
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
    add: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] += v; 
    },
    get_grad: function(x, y, d) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      return this.dw[ix]; 
    },
    set_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] = v; 
    },
    add_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] += v; 
    },
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
    setGradConst: function(a) { for(var k=0;k<this.dw.length;k++) { this.dw[k] = a; }},
    zero: function(){for(var k=0;k<this.w.length;k++) { this.dw[k] = 0.0; this.w[k] = 0.0;}},

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

      var n = this.sx*this.sy*this.depth;
      this.w = global.zeros(n);
      this.dw = global.zeros(n);
      // copy over the elements.
      for(var i=0;i<n;i++) {
        this.w[i] = json.w[i];
      }
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
  
  // concatinate v1.w with v2.w into 1D v3.w
  var concat_vol = function(v1, v2){
    var v3Len = v1.w.length + v2.w.length;
    var v3 = new Vol(1,1, v3Len, 0.0);
    for(var i = 0; i < v1.w.length; i++){
      v3.w[i] = v1.w[i];
    }
    for(var i = 0; i < v2.w.length; i++){
      v3.w[v1.w.length + i] = v2.w[i];
    }
    return v3;
  }
  
  global.concat_vol = concat_vol;
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
  // putting them together in one file because they are very similar
  var ConvLayer = function(opt) {
    var opt = opt || {};

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }
  ConvLayer.prototype = {
    forward: function(V, is_training) {
      // optimized code by @mdda that achieves 2x speedup over previous version

      this.in_act = V;
      var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
      
      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            var a = 0.0;
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
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

      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                    f.dw[ix2] += V.w[ix1]*chain_grad;
                    V.dw[ix1] += f.w[ix2]*chain_grad;
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
      json.pad = this.pad;
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
      this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
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
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
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
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

    // computed
    this.out_depth = this.in_depth;
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'pool';
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
  }

  PoolLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      var n=0; // a counter for switches
      for(var d=0;d<this.out_depth;d++) {
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            // convolve centered at this particular location
            var a = -99999; // hopefully small enough ;\
            var winx=-1,winy=-1;
            for(var fx=0;fx<this.sx;fx++) {
              for(var fy=0;fy<this.sy;fy++) {
                var oy = y+fy;
                var ox = x+fx;
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

      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {

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
      json.pad = this.pad;
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
      this.pad = typeof json.pad !== 'undefined' ? json.pad : 0; // backwards compatibility
      this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth); // need to re-init these appropriately
      this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    }
  }

  global.PoolLayer = PoolLayer;

})(convnetjs);


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
      if(!y.w){
        //y is a scaler label
        for(var i=0;i<N;i++) {
          indicator = i === y? this.max_val : this.min_val;
          //ideal output = max_val, dw should be negative
          //ideal output = min_val, dw should be positive
          if(indicator == this.max_val || this.out_act.w[i] == this.max_val){
            //it is a fired neuron
            this.in_act.dw[i] = (this.out_act.w[i] - indicator*1.1) * 3;
          }
          if(this.in_act.dw[i] != 0){
            cost++;
          }
        }
      }else{
        //y is a volume
        for(var i=0;i<N;i++) {
          indicator = (y.w[i] < this.threshold) ? this.min_val : this.max_val;
          if(indicator == this.max_val || this.out_act.w[i] == this.max_val){
            //it is a fired neuron
            this.in_act.dw[i] = (this.out_act.w[i] - indicator*1.1) * 3;
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
  }

  SVMLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      this.out_act = V; // nothing to do, output raw scores
      return V;
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
    }
  }
  
  global.RegressionLayer = RegressionLayer;
  global.SoftmaxLayer = SoftmaxLayer;
  global.SVMLayer = SVMLayer;
  global.BinaryReinforceLayer = BinaryReinforceLayer;
})(convnetjs);


(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  // Implements pull up layer (step function)
  // x -> min_val (default 0) or max_val (default 1) 
  // it simulate the output target cost function, and the output target depends on the sign of the bp error
  
  var StepLayer = function(opt){
    var opt = opt || {};
    
    // optional
    this.min_val = typeof opt.min_val !== 'undefined' ? opt.min_val : 0;
    this.max_val = typeof opt.max_val !== 'undefined' ? opt.max_val : 1.0;
    this.threshold = typeof opt.threshold !== 'undefined' ? opt.threshold : 0.3; //arbitrary magic number
    
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'step';
    this.init();
  }
  StepLayer.prototype = {
    init: function(){
      this.out_act = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
    },
    
    forward: function(V, is_training) {
      this.out_act.setConst(0.0);
      this.in_act = V;
      
      for(var i=0;i<this.in_act.w.length;i++) { 
        this.out_act.w[i] = (this.in_act.w[i] < this.threshold) ? this.min_val : this.max_val;
      }
      
      return this.out_act;
    },
    
    backward: function() {
      this.in_act.setGradConst(0.0);
      var indicator;
      
      for(var i=0;i<this.in_act.w.length;i++) {
        if(this.out_act.dw[i] != 0){
          indicator = (this.out_act.dw[i] > 0)? this.min_val : this.max_val;
          //ideal output = min_val, dw should be positive
          //ideal output = max_val, dw should be negative
          this.in_act.dw[i] = this.out_act.w[i] - indicator;
        }
      }
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
      
      //init
      this.init();
    }
  }
  
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

  // Implements Maxout nnonlinearity that computes
  // x -> max(x)
  // where x is a vector of size group_size. Ideally of course,
  // the input size should be exactly divisible by group_size
  var MaxoutLayer = function(opt) {
    var opt = opt || {};

    // required
    this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = Math.floor(opt.in_depth / this.group_size);
    this.layer_type = 'maxout';

    this.switches = global.zeros(this.out_sx*this.out_sy*this.out_depth); // useful for backprop
  }
  MaxoutLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var N = this.out_depth; 
      var V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

      // optimization branch. If we're operating on 1D arrays we dont have
      // to worry about keeping track of x,y,d coordinates inside
      // input volumes. In convnets we do :(
      if(this.out_sx === 1 && this.out_sy === 1) {
        for(var i=0;i<N;i++) {
          var ix = i * this.group_size; // base index offset
          var a = V.w[ix];
          var ai = 0;
          for(var j=1;j<this.group_size;j++) {
            var a2 = V.w[ix+j];
            if(a2 > a) {
              a = a2;
              ai = j;
            }
          }
          V2.w[i] = a;
          this.switches[i] = ix + ai;
        }
      } else {
        var n=0; // counter for switches
        for(var x=0;x<V.sx;x++) {
          for(var y=0;y<V.sy;y++) {
            for(var i=0;i<N;i++) {
              var ix = i * this.group_size;
              var a = V.get(x, y, ix);
              var ai = 0;
              for(var j=1;j<this.group_size;j++) {
                var a2 = V.get(x, y, ix+j);
                if(a2 > a) {
                  a = a2;
                  ai = j;
                }
              }
              V2.set(x,y,i,a);
              this.switches[n] = ix + ai;
              n++;
            }
          }
        }

      }
      this.out_act = V2;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act; // we need to set dw of this
      var V2 = this.out_act;
      var N = this.out_depth;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data

      // pass the gradient through the appropriate switch
      if(this.out_sx === 1 && this.out_sy === 1) {
        for(var i=0;i<N;i++) {
          var chain_grad = V2.dw[i];
          V.dw[this.switches[i]] = chain_grad;
        }
      } else {
        // bleh okay, lets do this the hard way
        var n=0; // counter for switches
        for(var x=0;x<V2.sx;x++) {
          for(var y=0;y<V2.sy;y++) {
            for(var i=0;i<N;i++) {
              var chain_grad = V2.get_grad(x,y,i);
              V.set_grad(x,y,this.switches[n],chain_grad);
              n++;
            }
          }
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
      json.group_size = this.group_size;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type; 
      this.group_size = json.group_size;
      this.switches = global.zeros(this.group_size);
    }
  }

  // a helper function, since tanh is not yet part of ECMAScript. Will be in v6.
  function tanh(x) {
    var y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
  }
  // Implements Tanh nnonlinearity elementwise
  // x -> tanh(x) 
  // so the output is between -1 and 1.
  var TanhLayer = function(opt) {
    var opt = opt || {};

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'tanh';
  }
  TanhLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var V2 = V.cloneAndZero();
      var N = V.w.length;
      for(var i=0;i<N;i++) { 
        V2.w[i] = tanh(V.w[i]);
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
        V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
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
  
  global.StepLayer = StepLayer;
  global.TanhLayer = TanhLayer;
  global.MaxoutLayer = MaxoutLayer;
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
  
  var BufferLayer = function(opt) {
      this.init(opt);
  }
  
  BufferLayer.prototype = {
    init : function(opt){
      var opt = opt || {};

      // required
      this.sx = opt.sx; // filter size
      this.in_depth = opt.in_depth;
      this.in_sx = opt.in_sx;
      this.in_sy = opt.in_sy;
  
      // optional
      this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
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
    
    purge: function(){
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
      json.sx = this.sx;
      json.sy = this.sy;
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
      this.sx = json.sx;
      this.sy = json.sy;
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
  
  // TODO
  // these two classes are not bug free
  // need to address the error bp through the recurrent connection by using the table
  //
  
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

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  var assert = global.assert;

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  var Net = function(options) {
    this.layers = [];
    this.bufferLayer = [];
  }

  Net.prototype = {
    // takes a list of layer definitions and creates the network layer objects
    makeLayers: function(defs) {

      // few checks
      assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
      assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

      // desugar layer_defs for adding activation, dropout layers etc
      var desugar = function() {
        var new_defs = [];
        for(var i=0;i<defs.length;i++) {
          var def = defs[i];
          
          if(def.type==='softmax' || def.type==='svm' || def.type==='binaryReinforce') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_classes});
          }

          if(def.type==='regression') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_neurons});
          }

          if((def.type==='fc' || def.type==='conv') 
              && typeof(def.bias_pref) === 'undefined'){
            def.bias_pref = 0.0;
            if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
              def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
              // otherwise it's technically possible that a relu unit will never turn on (by chance)
              // and will never get any gradient and never contribute any computation. Dead relu.
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
            else if (def.activation==='step') { new_defs.push({type:'step'}); }
            else if (def.activation==='tanh') { new_defs.push({type:'tanh'}); }
            else if (def.activation==='maxout') {
              // create maxout activation, and pass along group size, if provided
              var gs = def.group_size !== 'undefined' ? def.group_size : 2;
              new_defs.push({type:'maxout', group_size:gs});
            }
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
          case 'lstm' : this.layers.push(new global.LSTMLayer(def)); break;
          case 'buffer' : 
            var bufferLayer = new global.BufferLayer(def);
            this.layers.push(bufferLayer); 
            this.bufferLayer.push(bufferLayer)
          break;
          case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
          case 'input': this.layers.push(new global.InputLayer(def)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
          case 'binaryReinforce': this.layers.push(new global.BinaryReinforceLayer(def)); break;
          case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
          case 'conv': this.layers.push(new global.ConvLayer(def)); break;
          case 'pool': this.layers.push(new global.PoolLayer(def)); break;
          case 'relu': this.layers.push(new global.ReluLayer(def)); break;
          case 'step': this.layers.push(new global.StepLayer(def)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
          case 'tanh': this.layers.push(new global.TanhLayer(def)); break;
          case 'maxout': this.layers.push(new global.MaxoutLayer(def)); break;
          case 'svm': this.layers.push(new global.SVMLayer(def)); break;
          default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
        }
      }
    },

    // forward prop the network. 
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    forward: function(V, is_training) {
      if(typeof(is_training) === 'undefined') is_training = false;
      var act = this.layers[0].forward(V, is_training);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
      }
      return act;
    },

    getCostLoss: function(V, y) {
      this.forward(V, false);
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y);
      return loss;
    },
    
    reset: function(){
      for(var i=0;i<this.layers.length;i++) {
        if(typeof this.layers[i].reset === "function"){
          this.layers[i].reset();
        }
      }
    },
    
    // backprop: compute gradients wrt all parameters
    backward: function(y) {
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
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
      // this is a convenience function for returning the argmax
      // prediction, assuming the last layer of the net is a softmax
      var S = this.layers[this.layers.length-1];
      assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

      var p = S.out_act.w;
      var maxv = p[0];
      var maxi = 0;
      for(var i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i;}
      }
      return maxi; // return index of the class with highest class probability
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
        if(t==='tanh') { L = new global.TanhLayer(); }
        if(t==='dropout') { L = new global.DropoutLayer(); }
        if(t==='conv') { L = new global.ConvLayer(); }
        if(t==='pool') { L = new global.PoolLayer(); }
        if(t==='lrn') { L = new global.LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new global.SoftmaxLayer(); }
        if(t==='regression') { L = new global.RegressionLayer(); }
        if(t==='fc') { L = new global.FullyConnLayer(); }
        if(t==='maxout') { L = new global.MaxoutLayer(); }
        if(t==='svm') { L = new global.SVMLayer(); }
        if(t==='binaryReinforce') { L = new global.BinaryReinforceLayer(); }
        if(t==='lstm') { L = new global.LSTMLayer(); }
        if(t==='buffer') { L = new global.BufferLayer(); }
        if(t==='step') { L = new global.StepLayer(); }
        
        L.fromJSON(Lj);
        this.layers.push(L);
      }
    }
  }
  
  global.Net = Net;
})(convnetjs);

var Buckets = require('buckets-js');

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  /**
   * mode:
   *    normal    : classic training method
   *    hardcore  : keep training on the failed ones 10 times or until cost = 0
   *    
   */
  var Trainer = function(net, options) {

    this.net = net;

    var options = options || {};
    // enhanced params
    this.mode = typeof options.mode !== 'undefined' ? options.mode : "normal";
    this.costThreshold = typeof options.costThreshold !== 'undefined' ? options.costThreshold : 0;
    
    // basic training params
    this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
    this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
    this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
    this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
    this.method = typeof options.method !== 'undefined' ? options.method : 'sgd'; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

    this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;
    this.ro = typeof options.ro !== 'undefined' ? options.ro : 0.95; // used in adadelta
    this.eps = typeof options.eps !== 'undefined' ? options.eps : 1e-8; // used in adam or adadelta
    this.beta1 = typeof options.beta1 !== 'undefined' ? options.beta1 : 0.9; // used in adam
    this.beta2 = typeof options.beta2 !== 'undefined' ? options.beta2 : 0.999; // used in adam

    this.k = 0; // iteration counter
    this.gsum = []; // last iteration gradients (used for momentum calculations)
    this.xsum = []; // used in adam or adadelta
    
    // constant
    this.maxPQueueSize = 5;
    this.maxPQueueRetry = 20;
    
    // init
    this.costPQueue =  new Buckets.PriorityQueue(function(sampleA, sampleB){
      if(sampleA.priority < sampleB.priority){
        return -1;
      }else if(sampleA.priority > sampleB.priority){
        return 1;
      }else{
        return 0;
      }
    });
  }

  Trainer.prototype = {
    train: function(x, y) {
      var trainStatus = this.trainCore(x,y);
      
      if(this.mode == "hardcore"){
        var cost = trainStatus.cost_loss;
        
        if(cost > this.costThreshold && this.costPQueue.size() <= this.maxPQueueSize){
          this.costPQueue.add({data:{x:x, y:y}, priority:cost}); //using inverse cost
        }
        
        var iter = 0;
        while(!this.costPQueue.isEmpty() && iter<this.maxPQueueRetry){
          var trainingPair = this.costPQueue.dequeue().data;
          trainStatus = this.trainCore(trainingPair.x, trainingPair.y);
          cost = trainStatus.cost_loss;
          if(cost > this.costThreshold){
            // push the sample back
            this.costPQueue.add({data:trainingPair, priority:cost});
          }
          iter++;
        }
        
      }
      
      return trainStatus; // return the highest cost status
    },
    
    
    trainCore: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        var pglist = this.net.getParamsAndGrads();

        // initialize lists for accumulators. Will only be done once on first iteration
        if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
          // only vanilla sgd doesnt need either lists
          // momentum needs gsum
          // adagrad needs gsum
          // adam and adadelta needs gsum and xsum
          for(var i=0;i<pglist.length;i++) {
            this.gsum.push(global.zeros(pglist[i].params.length));
            if(this.method === 'adam' || this.method === 'adadelta') {
              this.xsum.push(global.zeros(pglist[i].params.length));
            } else {
              this.xsum.push([]); // conserve memory
            }
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

            var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

            var gsumi = this.gsum[i];
            var xsumi = this.xsum[i];
            if(this.method === 'adam') {
              // adam update
              gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
              xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
              var biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
              var biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
              var dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
              p[j] += dx;
            } else if(this.method === 'adagrad') {
              // adagrad update
              gsumi[j] = gsumi[j] + gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
              p[j] += dx;
            } else if(this.method === 'windowgrad') {
              // this is adagrad but with a moving window weighted average
              // so the gradient is not accumulated over the entire history of the run. 
              // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
              p[j] += dx;
            } else if(this.method === 'adadelta') {
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
              xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
              p[j] += dx;
            } else if(this.method === 'nesterov') {
            	var dx = gsumi[j];
            	gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                p[j] += dx;
            } else {
              // assume SGD
              if(this.momentum > 0.0) {
                // momentum update
                var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                gsumi[j] = dx; // back this up for next iteration of momentum
                p[j] += dx; // apply corrected gradient
              } else {
                // vanilla sgd
                p[j] +=  - this.learning_rate * gij;
              }
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
  
  global.Trainer = Trainer;
  global.SGDTrainer = Trainer; // backwards compatibility
})(convnetjs);


(function(global) {
  "use strict";

  // used utilities, make explicit local references
  var randf = global.randf;
  var randi = global.randi;
  var Net = global.Net;
  var Trainer = global.Trainer;
  var maxmin = global.maxmin;
  var randperm = global.randperm;
  var weightedSample = global.weightedSample;
  var getopt = global.getopt;
  var arrUnique = global.arrUnique;

  /*
  A MagicNet takes data: a list of convnetjs.Vol(), and labels
  which for now are assumed to be class indeces 0..K. MagicNet then:
  - creates data folds for cross-validation
  - samples candidate networks
  - evaluates candidate networks on all data folds
  - produces predictions by model-averaging the best networks
  */
  var MagicNet = function(data, labels, opt) {
    var opt = opt || {};
    if(typeof data === 'undefined') { data = []; }
    if(typeof labels === 'undefined') { labels = []; }

    // required inputs
    this.data = data; // store these pointers to data
    this.labels = labels;

    // optional inputs
    this.train_ratio = getopt(opt, 'train_ratio', 0.7);
    this.num_folds = getopt(opt, 'num_folds', 10);
    this.num_candidates = getopt(opt, 'num_candidates', 50); // we evaluate several in parallel
    // how many epochs of data to train every network? for every fold?
    // higher values mean higher accuracy in final results, but more expensive
    this.num_epochs = getopt(opt, 'num_epochs', 50); 
    // number of best models to average during prediction. Usually higher = better
    this.ensemble_size = getopt(opt, 'ensemble_size', 10);

    // candidate parameters
    this.batch_size_min = getopt(opt, 'batch_size_min', 10);
    this.batch_size_max = getopt(opt, 'batch_size_max', 300);
    this.l2_decay_min = getopt(opt, 'l2_decay_min', -4);
    this.l2_decay_max = getopt(opt, 'l2_decay_max', 2);
    this.learning_rate_min = getopt(opt, 'learning_rate_min', -4);
    this.learning_rate_max = getopt(opt, 'learning_rate_max', 0);
    this.momentum_min = getopt(opt, 'momentum_min', 0.9);
    this.momentum_max = getopt(opt, 'momentum_max', 0.9);
    this.neurons_min = getopt(opt, 'neurons_min', 5);
    this.neurons_max = getopt(opt, 'neurons_max', 30);

    // computed
    this.folds = []; // data fold indices, gets filled by sampleFolds()
    this.candidates = []; // candidate networks that are being currently evaluated
    this.evaluated_candidates = []; // history of all candidates that were fully evaluated on all folds
    this.unique_labels = arrUnique(labels);
    this.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
    this.foldix = 0; // index of active fold

    // callbacks
    this.finish_fold_callback = null;
    this.finish_batch_callback = null;

    // initializations
    if(this.data.length > 0) {
      this.sampleFolds();
      this.sampleCandidates();
    }
  };

  MagicNet.prototype = {

    // sets this.folds to a sampling of this.num_folds folds
    sampleFolds: function() {
      var N = this.data.length;
      var num_train = Math.floor(this.train_ratio * N);
      this.folds = []; // flush folds, if any
      for(var i=0;i<this.num_folds;i++) {
        var p = randperm(N);
        this.folds.push({train_ix: p.slice(0, num_train), test_ix: p.slice(num_train, N)});
      }
    },

    // returns a random candidate network
    sampleCandidate: function() {
      var input_depth = this.data[0].w.length;
      var num_classes = this.unique_labels.length;

      // sample network topology and hyperparameters
      var layer_defs = [];
      layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: input_depth});
      var nl = weightedSample([0,1,2,3], [0.2, 0.3, 0.3, 0.2]); // prefer nets with 1,2 hidden layers
      for(var q=0;q<nl;q++) {
        var ni = randi(this.neurons_min, this.neurons_max);
        var act = ['tanh','maxout','relu'][randi(0,3)];
        if(randf(0,1)<0.5) {
          var dp = Math.random();
          layer_defs.push({type:'fc', num_neurons: ni, activation: act, drop_prob: dp});
        } else {
          layer_defs.push({type:'fc', num_neurons: ni, activation: act});
        }
      }
      layer_defs.push({type:'softmax', num_classes: num_classes});
      var net = new Net();
      net.makeLayers(layer_defs);

      // sample training hyperparameters
      var bs = randi(this.batch_size_min, this.batch_size_max); // batch size
      var l2 = Math.pow(10, randf(this.l2_decay_min, this.l2_decay_max)); // l2 weight decay
      var lr = Math.pow(10, randf(this.learning_rate_min, this.learning_rate_max)); // learning rate
      var mom = randf(this.momentum_min, this.momentum_max); // momentum. Lets just use 0.9, works okay usually ;p
      var tp = randf(0,1); // trainer type
      var trainer_def;
      if(tp<0.33) {
        trainer_def = {method:'adadelta', batch_size:bs, l2_decay:l2};
      } else if(tp<0.66) {
        trainer_def = {method:'adagrad', learning_rate: lr, batch_size:bs, l2_decay:l2};
      } else {
        trainer_def = {method:'sgd', learning_rate: lr, momentum: mom, batch_size:bs, l2_decay:l2};
      }
      
      var trainer = new Trainer(net, trainer_def);

      var cand = {};
      cand.acc = [];
      cand.accv = 0; // this will maintained as sum(acc) for convenience
      cand.layer_defs = layer_defs;
      cand.trainer_def = trainer_def;
      cand.net = net;
      cand.trainer = trainer;
      return cand;
    },

    // sets this.candidates with this.num_candidates candidate nets
    sampleCandidates: function() {
      this.candidates = []; // flush, if any
      for(var i=0;i<this.num_candidates;i++) {
        var cand = this.sampleCandidate();
        this.candidates.push(cand);
      }
    },

    step: function() {
      
      // run an example through current candidate
      this.iter++;

      // step all candidates on a random data point
      var fold = this.folds[this.foldix]; // active fold
      var dataix = fold.train_ix[randi(0, fold.train_ix.length)];
      for(var k=0;k<this.candidates.length;k++) {
        var x = this.data[dataix];
        var l = this.labels[dataix];
        this.candidates[k].trainer.train(x, l);
      }

      // process consequences: sample new folds, or candidates
      var lastiter = this.num_epochs * fold.train_ix.length;
      if(this.iter >= lastiter) {
        // finished evaluation of this fold. Get final validation
        // accuracies, record them, and go on to next fold.
        var val_acc = this.evalValErrors();
        for(var k=0;k<this.candidates.length;k++) {
          var c = this.candidates[k];
          c.acc.push(val_acc[k]);
          c.accv += val_acc[k];
        }
        this.iter = 0; // reset step number
        this.foldix++; // increment fold

        if(this.finish_fold_callback !== null) {
          this.finish_fold_callback();
        }

        if(this.foldix >= this.folds.length) {
          // we finished all folds as well! Record these candidates
          // and sample new ones to evaluate.
          for(var k=0;k<this.candidates.length;k++) {
            this.evaluated_candidates.push(this.candidates[k]);
          }
          // sort evaluated candidates according to accuracy achieved
          this.evaluated_candidates.sort(function(a, b) { 
            return (a.accv / a.acc.length) 
                 > (b.accv / b.acc.length) 
                 ? -1 : 1;
          });
          // and clip only to the top few ones (lets place limit at 3*ensemble_size)
          // otherwise there are concerns with keeping these all in memory 
          // if MagicNet is being evaluated for a very long time
          if(this.evaluated_candidates.length > 3 * this.ensemble_size) {
            this.evaluated_candidates = this.evaluated_candidates.slice(0, 3 * this.ensemble_size);
          }
          if(this.finish_batch_callback !== null) {
            this.finish_batch_callback();
          }
          this.sampleCandidates(); // begin with new candidates
          this.foldix = 0; // reset this
        } else {
          // we will go on to another fold. reset all candidates nets
          for(var k=0;k<this.candidates.length;k++) {
            var c = this.candidates[k];
            var net = new Net();
            net.makeLayers(c.layer_defs);
            var trainer = new Trainer(net, c.trainer_def);
            c.net = net;
            c.trainer = trainer;
          }
        }
      }
    },

    evalValErrors: function() {
      // evaluate candidates on validation data and return performance of current networks
      // as simple list
      var vals = [];
      var fold = this.folds[this.foldix]; // active fold
      for(var k=0;k<this.candidates.length;k++) {
        var net = this.candidates[k].net;
        var v = 0.0;
        for(var q=0;q<fold.test_ix.length;q++) {
          var x = this.data[fold.test_ix[q]];
          var l = this.labels[fold.test_ix[q]];
          net.forward(x);
          var yhat = net.getPrediction();
          v += (yhat === l ? 1.0 : 0.0); // 0 1 loss
        }
        v /= fold.test_ix.length; // normalize
        vals.push(v);
      }
      return vals;
    },

    // returns prediction scores for given test data point, as Vol
    // uses an averaged prediction from the best ensemble_size models
    // x is a Vol.
    predict_soft: function(data) {
      // forward prop the best networks
      // and accumulate probabilities at last layer into a an output Vol

      var eval_candidates = [];
      var nv = 0;
      if(this.evaluated_candidates.length === 0) {
        // not sure what to do here, first batch of nets hasnt evaluated yet
        // lets just predict with current candidates.
        nv = this.candidates.length;
        eval_candidates = this.candidates;
      } else {
        // forward prop the best networks from evaluated_candidates
        nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
        eval_candidates = this.evaluated_candidates
      }

      // forward nets of all candidates and average the predictions
      var xout, n;
      for(var j=0;j<nv;j++) {
        var net = eval_candidates[j].net;
        var x = net.forward(data);
        if(j===0) { 
          xout = x; 
          n = x.w.length; 
        } else {
          // add it on
          for(var d=0;d<n;d++) {
            xout.w[d] += x.w[d];
          }
        }
      }
      // produce average
      for(var d=0;d<n;d++) {
        xout.w[d] /= nv;
      }
      return xout;
    },

    predict: function(data) {
      var xout = this.predict_soft(data);
      if(xout.w.length !== 0) {
        var stats = maxmin(xout.w);
        var predicted_label = stats.maxi; 
      } else {
        var predicted_label = -1; // error out
      }
      return predicted_label;

    },

    toJSON: function() {
      // dump the top ensemble_size networks as a list
      var nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
      var json = {};
      json.nets = [];
      for(var i=0;i<nv;i++) {
        json.nets.push(this.evaluated_candidates[i].net.toJSON());
      }
      return json;
    },

    fromJSON: function(json) {
      this.ensemble_size = json.nets.length;
      this.evaluated_candidates = [];
      for(var i=0;i<this.ensemble_size;i++) {
        var net = new Net();
        net.fromJSON(json.nets[i]);
        var dummy_candidate = {};
        dummy_candidate.net = net;
        this.evaluated_candidates.push(dummy_candidate);
      }
    },

    // callback functions
    // called when a fold is finished, while evaluating a batch
    onFinishFold: function(f) { this.finish_fold_callback = f; },
    // called when a batch of candidates has finished evaluating
    onFinishBatch: function(f) { this.finish_batch_callback = f; }
    
  };

  global.MagicNet = MagicNet;
})(convnetjs);

(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.convnetjs = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(convnetjs);


},{"buckets-js":3}],2:[function(require,module,exports){
var convnetjs = require('./build/convnet_pure');
window.convnetjs = convnetjs;
},{"./build/convnet_pure":1}],3:[function(require,module,exports){
// buckets 1.90.0 
// (c) 2013, 2015 Mauricio Santos <mauriciosantoss@gmail.com> 
// https://github.com/mauriciosantos/Buckets-JS

(function (root, factory) {
    // UMD (Universal Module Definition) https://github.com/umdjs/umd

    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module unless amdModuleId is set
        define([], factory);
    } else if (typeof exports === 'object') {
        // Node. Does not work with strict CommonJS, but
        // only CommonJS-like environments that support module.exports,
        // like Node.
        module.exports = factory();
    } else {
        // Browser globals (root is window)
        root.buckets = factory();
    }

}(this, function () {

    'use strict';

    /**
     * Top level namespace for Buckets,
     * a JavaScript data structure library.
     * @name buckets
     */
    var buckets = {};

    /**
     * Default function to compare element order.
     * @function
     * @private
     */
    buckets.defaultCompare = function (a, b) {
        if (a < b) {
            return -1;
        }
        if (a === b) {
            return 0;
        }
        return 1;
    };

    /**
     * Default function to test equality.
     * @function
     * @private
     */
    buckets.defaultEquals = function (a, b) {
        return a === b;
    };

    /**
     * Default function to convert an object to a string.
     * @function
     * @private
     */
    buckets.defaultToString = function (item) {
        if (item === null) {
            return 'BUCKETS_NULL';
        }
        if (buckets.isUndefined(item)) {
            return 'BUCKETS_UNDEFINED';
        }
        if (buckets.isString(item)) {
            return item;
        }
        return item.toString();
    };

    /**
     * Checks if the given argument is a function.
     * @function
     * @private
     */
    buckets.isFunction = function (func) {
        return (typeof func) === 'function';
    };

    /**
     * Checks if the given argument is undefined.
     * @function
     * @private
     */
    buckets.isUndefined = function (obj) {
        return obj === undefined;
    };

    /**
     * Checks if the given argument is a string.
     * @function
     * @private
     */
    buckets.isString = function (obj) {
        return Object.prototype.toString.call(obj) === '[object String]';
    };

    /**
     * Reverses a compare function.
     * @function
     * @private
     */
    buckets.reverseCompareFunction = function (compareFunction) {
        if (!buckets.isFunction(compareFunction)) {
            return function (a, b) {
                if (a < b) {
                    return 1;
                }
                if (a === b) {
                    return 0;
                }
                return -1;
            };
        }
        return function (d, v) {
            return compareFunction(d, v) * -1;
        };

    };

    /**
     * Returns an equal function given a compare function.
     * @function
     * @private
     */
    buckets.compareToEquals = function (compareFunction) {
        return function (a, b) {
            return compareFunction(a, b) === 0;
        };
    };


    /**
     * @namespace Contains various functions for manipulating arrays.
     */
    buckets.arrays = {};

    /**
     * Returns the index of the first occurrence of the specified item
     * within the specified array.
     * @param {*} array The array.
     * @param {*} item The element to search for.
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {number} The index of the first occurrence of the specified element
     * or -1 if not found.
     */
    buckets.arrays.indexOf = function (array, item, equalsFunction) {
        var equals = equalsFunction || buckets.defaultEquals,
            length = array.length,
            i;
        for (i = 0; i < length; i += 1) {
            if (equals(array[i], item)) {
                return i;
            }
        }
        return -1;
    };

    /**
     * Returns the index of the last occurrence of the specified element
     * within the specified array.
     * @param {*} array The array.
     * @param {Object} item The element to search for.
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {number} The index of the last occurrence of the specified element
     * within the specified array or -1 if not found.
     */
    buckets.arrays.lastIndexOf = function (array, item, equalsFunction) {
        var equals = equalsFunction || buckets.defaultEquals,
            length = array.length,
            i;
        for (i = length - 1; i >= 0; i -= 1) {
            if (equals(array[i], item)) {
                return i;
            }
        }
        return -1;
    };

    /**
     * Returns true if the array contains the specified element.
     * @param {*} array The array.
     * @param {Object} item The element to search for.
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {boolean} True if the specified array contains the specified element.
     */
    buckets.arrays.contains = function (array, item, equalsFunction) {
        return buckets.arrays.indexOf(array, item, equalsFunction) >= 0;
    };

    /**
     * Removes the first ocurrence of the specified element from the specified array.
     * @param {*} array The array.
     * @param {*} item The element to remove.
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {boolean} True If the array changed after this call.
     */
    buckets.arrays.remove = function (array, item, equalsFunction) {
        var index = buckets.arrays.indexOf(array, item, equalsFunction);
        if (index < 0) {
            return false;
        }
        array.splice(index, 1);
        return true;
    };

    /**
     * Returns the number of elements in the array equal
     * to the specified element.
     * @param {Array} array The array.
     * @param {Object} item The element.
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {number} The number of elements in the specified array.
     * equal to the specified item.
     */
    buckets.arrays.frequency = function (array, item, equalsFunction) {
        var equals = equalsFunction || buckets.defaultEquals,
            length = array.length,
            freq = 0,
            i;
        for (i = 0; i < length; i += 1) {
            if (equals(array[i], item)) {
                freq += 1;
            }
        }
        return freq;
    };

    /**
     * Returns true if the provided arrays are equal.
     * Two arrays are considered equal if both contain the same number
     * of elements and all corresponding pairs of elements
     * are equal and are in the same order.
     * @param {Array} array1
     * @param {Array} array2
     * @param {function(Object,Object):boolean=} equalsFunction Optional function to
     * check equality between two elements. Receives two arguments and returns true if they are equal.
     * @return {boolean} True if the two arrays are equal.
     */
    buckets.arrays.equals = function (array1, array2, equalsFunction) {
        var equals = equalsFunction || buckets.defaultEquals,
            length = array1.length,
            i;

        if (array1.length !== array2.length) {
            return false;
        }
        for (i = 0; i < length; i += 1) {
            if (!equals(array1[i], array2[i])) {
                return false;
            }
        }
        return true;
    };

    /**
     * Returns a shallow copy of the specified array.
     * @param {*} array The array to copy.
     * @return {Array} A copy of the specified array.
     */
    buckets.arrays.copy = function (array) {
        return array.concat();
    };

    /**
     * Swaps the elements at the specified positions in the specified array.
     * @param {Array} array The array.
     * @param {number} i The index of the first element.
     * @param {number} j The index of second element.
     * @return {boolean} True if the array is defined and the indexes are valid.
     */
    buckets.arrays.swap = function (array, i, j) {
        var temp;

        if (i < 0 || i >= array.length || j < 0 || j >= array.length) {
            return false;
        }
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
        return true;
    };

    /**
     * Executes the provided function once per element present in the array.
     * @param {Array} array The array.
     * @param {function(Object):*} callback Function to execute,
     * invoked with an element as argument. To break the iteration you can
     * optionally return false in the callback.
     */
    buckets.arrays.forEach = function (array, callback) {
        var lenght = array.length,
            i;
        for (i = 0; i < lenght; i += 1) {
            if (callback(array[i]) === false) {
                return;
            }
        }
    };


    /**
     * Creates an empty bag.
     * @class <p>A bag is a special kind of set in which members are
     * allowed to appear more than once.</p>
     * <p>If the inserted elements are custom objects, a function
     * that maps elements to unique strings must be provided at construction time.</p>
     * <p>Example:</p>
     * <pre>
     * function petToUniqueString(pet) {
     *  return pet.type + ' ' + pet.name;
     * }
     * </pre>
     *
     * @constructor
     * @param {function(Object):string=} toStrFunction Optional function
     * to convert elements to unique strings. If the elements aren't strings or if toString()
     * is not appropriate, a custom function which receives an object and returns a
     * unique string must be provided.
     */
    buckets.Bag = function (toStrFunction) {

        /** 
         * @exports bag as buckets.Bag
         * @private
         */
        var bag = {},
            // Function to convert elements to unique strings.
            toStrF = toStrFunction || buckets.defaultToString,
            // Underlying  Storage
            dictionary = new buckets.Dictionary(toStrF),
            // Number of elements in the bag, including duplicates.
            nElements = 0;
        /**
         * Adds nCopies of the specified element to the bag.
         * @param {Object} element Element to add.
         * @param {number=} nCopies The number of copies to add, if this argument is
         * undefined 1 copy is added.
         * @return {boolean} True unless element is undefined.
         */
        bag.add = function (element, nCopies) {
            var node;
            if (isNaN(nCopies) || buckets.isUndefined(nCopies)) {
                nCopies = 1;
            }
            if (buckets.isUndefined(element) || nCopies <= 0) {
                return false;
            }

            if (!bag.contains(element)) {
                node = {
                    value: element,
                    copies: nCopies
                };
                dictionary.set(element, node);
            } else {
                dictionary.get(element).copies += nCopies;
            }
            nElements += nCopies;
            return true;
        };

        /**
         * Counts the number of copies of the specified element in the bag.
         * @param {Object} element The element to search for.
         * @return {number} The number of copies of the element, 0 if not found.
         */
        bag.count = function (element) {
            if (!bag.contains(element)) {
                return 0;
            }
            return dictionary.get(element).copies;
        };

        /**
         * Returns true if the bag contains the specified element.
         * @param {Object} element Element to search for.
         * @return {boolean} True if the bag contains the specified element,
         * false otherwise.
         */
        bag.contains = function (element) {
            return dictionary.containsKey(element);
        };

        /**
         * Removes nCopies of the specified element from the bag.
         * If the number of copies to remove is greater than the actual number
         * of copies in the bag, all copies are removed.
         * @param {Object} element Element to remove.
         * @param {number=} nCopies The number of copies to remove, if this argument is
         * undefined 1 copy is removed.
         * @return {boolean} True if at least 1 copy was removed.
         */
        bag.remove = function (element, nCopies) {
            var node;
            if (isNaN(nCopies) || buckets.isUndefined(nCopies)) {
                nCopies = 1;
            }
            if (buckets.isUndefined(element) || nCopies <= 0) {
                return false;
            }

            if (!bag.contains(element)) {
                return false;
            }
            node = dictionary.get(element);
            if (nCopies > node.copies) {
                nElements -= node.copies;
            } else {
                nElements -= nCopies;
            }
            node.copies -= nCopies;
            if (node.copies <= 0) {
                dictionary.remove(element);
            }
            return true;
        };

        /**
         * Returns an array containing all the elements in the bag in no particular order,
         * including multiple copies.
         * @return {Array} An array containing all the elements in the bag.
         */
        bag.toArray = function () {
            var a = [],
                values = dictionary.values(),
                vl = values.length,
                node,
                element,
                copies,
                i,
                j;
            for (i = 0; i < vl; i += 1) {
                node = values[i];
                element = node.value;
                copies = node.copies;
                for (j = 0; j < copies; j += 1) {
                    a.push(element);
                }
            }
            return a;
        };

        /**
         * Returns a set of unique elements in the bag.
         * @return {buckets.Set} A set of unique elements in the bag.
         */
        bag.toSet = function () {
            var set = new buckets.Set(toStrF),
                elements = dictionary.values(),
                l = elements.length,
                i;
            for (i = 0; i < l; i += 1) {
                set.add(elements[i].value);
            }
            return set;
        };

        /**
         * Executes the provided function once per element
         * present in the bag, including multiple copies.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked with an element as argument. To break the iteration you can
         * optionally return false in the callback.
         */
        bag.forEach = function (callback) {
            dictionary.forEach(function (k, v) {
                var value = v.value,
                    copies = v.copies,
                    i;
                for (i = 0; i < copies; i += 1) {
                    if (callback(value) === false) {
                        return false;
                    }
                }
                return true;
            });
        };
        /**
         * Returns the number of elements in the bag, including duplicates.
         * @return {number} The number of elements in the bag.
         */
        bag.size = function () {
            return nElements;
        };

        /**
         * Returns true if the bag contains no elements.
         * @return {boolean} True if the bag contains no elements.
         */
        bag.isEmpty = function () {
            return nElements === 0;
        };

        /**
         * Removes all the elements from the bag.
         */
        bag.clear = function () {
            nElements = 0;
            dictionary.clear();
        };

        /**
         * Returns true if the bag is equal to another bag.
         * Two bags are equal if they have the same elements and
         * same number of copies per element.
         * @param {buckets.Bag} other The other bag.
         * @return {boolean} True if the bag is equal to the given bag.
         */
        bag.equals = function (other) {
            var isEqual;
            if (buckets.isUndefined(other) || typeof other.toSet !== 'function') {
                return false;
            }
            if (bag.size() !== other.size()) {
                return false;
            }

            isEqual = true;
            other.forEach(function (element) {
                isEqual = (bag.count(element) === other.count(element));
                return isEqual;
            });
            return isEqual;
        };

        return bag;
    };


    /**
     * Creates an empty binary search tree.
     * @class <p> Binary search trees keep their elements in sorted order, so that 
     * lookup and other operations can use the principle of binary search. In a BST
     * the element in any node is larger than the elements in the node's
     * left sub-tree and smaller than the elements in the node's right sub-tree.</p>
     * <p>If the inserted elements are custom objects, a compare function must
     * be provided at construction time, otherwise the <=, === and >= operators are
     * used to compare elements.</p>
     * <p>Example:</p>
     * <pre>
     * function compare(a, b) {
     *  if (a is less than b by some ordering criterion) {
     *     return -1;
     *  } if (a is greater than b by the ordering criterion) {
     *     return 1;
     *  }
     *  // a must be equal to b
     *  return 0;
     * }
     * </pre>
     * @constructor
     * @param {function(Object,Object):number=} compareFunction Optional
     * function used to compare two elements. Must return a negative integer,
     * zero, or a positive integer as the first argument is less than, equal to,
     * or greater than the second.
     */
    buckets.BSTree = function (compareFunction) {

        /** 
         * @exports tree as buckets.BSTree
         * @private
         */
        var tree = {},
            // Function to compare elements.
            compare = compareFunction || buckets.defaultCompare,
            // Number of elements in the tree.
            nElements = 0,
            // The root node of the tree.
            root;

        // Returns the sub-node containing the specified element or undefined.
        function searchNode(root, element) {
            var node = root,
                cmp;
            while (node !== undefined && cmp !== 0) {
                cmp = compare(element, node.element);
                if (cmp < 0) {
                    node = node.leftCh;
                } else if (cmp > 0) {
                    node = node.rightCh;
                }
            }
            return node;
        }

        // Returns the sub-node containing the minimum element or undefined.
        function minimumAux(root) {
            var node = root;
            while (node.leftCh !== undefined) {
                node = node.leftCh;
            }
            return node;
        }

        /**
         * Inserts the specified element into the tree if it's not already present.
         * @param {Object} element The element to insert.
         * @return {boolean} True if the tree didn't already contain the element.
         */
        tree.add = function (element) {
            if (buckets.isUndefined(element)) {
                return false;
            }

            /**
             * @private
             */
            function insertNode(node) {
                var position = root,
                    parent,
                    cmp;

                while (position !== undefined) {
                    cmp = compare(node.element, position.element);
                    if (cmp === 0) {
                        return undefined;
                    }
                    if (cmp < 0) {
                        parent = position;
                        position = position.leftCh;
                    } else {
                        parent = position;
                        position = position.rightCh;
                    }
                }
                node.parent = parent;
                if (parent === undefined) {
                    // tree is empty
                    root = node;
                } else if (compare(node.element, parent.element) < 0) {
                    parent.leftCh = node;
                } else {
                    parent.rightCh = node;
                }
                return node;
            }

            var node = {
                element: element,
                leftCh: undefined,
                rightCh: undefined,
                parent: undefined
            };
            if (insertNode(node) !== undefined) {
                nElements += 1;
                return true;
            }
            return false;
        };

        /**
         * Removes all the elements from the tree.
         */
        tree.clear = function () {
            root = undefined;
            nElements = 0;
        };

        /**
         * Returns true if the tree contains no elements.
         * @return {boolean} True if the tree contains no elements.
         */
        tree.isEmpty = function () {
            return nElements === 0;
        };

        /**
         * Returns the number of elements in the tree.
         * @return {number} The number of elements in the tree.
         */
        tree.size = function () {
            return nElements;
        };

        /**
         * Returns true if the tree contains the specified element.
         * @param {Object} element Element to search for.
         * @return {boolean} True if the tree contains the element,
         * false otherwise.
         */
        tree.contains = function (element) {
            if (buckets.isUndefined(element)) {
                return false;
            }
            return searchNode(root, element) !== undefined;
        };

        /**
         * Removes the specified element from the tree.
         * @return {boolean} True if the tree contained the specified element.
         */
        tree.remove = function (element) {
            var node;

            function transplant(n1, n2) {
                if (n1.parent === undefined) {
                    root = n2;
                } else if (n1 === n1.parent.leftCh) {
                    n1.parent.leftCh = n2;
                } else {
                    n1.parent.rightCh = n2;
                }
                if (n2 !== undefined) {
                    n2.parent = n1.parent;
                }
            }

            function removeNode(node) {
                if (node.leftCh === undefined) {
                    transplant(node, node.rightCh);
                } else if (node.rightCh === undefined) {
                    transplant(node, node.leftCh);
                } else {
                    var y = minimumAux(node.rightCh);
                    if (y.parent !== node) {
                        transplant(y, y.rightCh);
                        y.rightCh = node.rightCh;
                        y.rightCh.parent = y;
                    }
                    transplant(node, y);
                    y.leftCh = node.leftCh;
                    y.leftCh.parent = y;
                }
            }

            node = searchNode(root, element);
            if (node === undefined) {
                return false;
            }
            removeNode(node);
            nElements -= 1;
            return true;
        };

        /**
         * Executes the provided function once per element present in the tree in in-order.
         * @param {function(Object):*} callback Function to execute, invoked with an element as 
         * argument. To break the iteration you can optionally return false in the callback.
         */
        tree.inorderTraversal = function (callback) {

            function inorderRecursive(node, callback, signal) {
                if (node === undefined || signal.stop) {
                    return;
                }
                inorderRecursive(node.leftCh, callback, signal);
                if (signal.stop) {
                    return;
                }
                signal.stop = callback(node.element) === false;
                if (signal.stop) {
                    return;
                }
                inorderRecursive(node.rightCh, callback, signal);
            }

            inorderRecursive(root, callback, {
                stop: false
            });
        };

        /**
         * Executes the provided function once per element present in the tree in pre-order.
         * @param {function(Object):*} callback Function to execute, invoked with an element as 
         * argument. To break the iteration you can optionally return false in the callback.
         */
        tree.preorderTraversal = function (callback) {

            function preorderRecursive(node, callback, signal) {
                if (node === undefined || signal.stop) {
                    return;
                }
                signal.stop = callback(node.element) === false;
                if (signal.stop) {
                    return;
                }
                preorderRecursive(node.leftCh, callback, signal);
                if (signal.stop) {
                    return;
                }
                preorderRecursive(node.rightCh, callback, signal);
            }

            preorderRecursive(root, callback, {
                stop: false
            });
        };

        /**
         * Executes the provided function once per element present in the tree in post-order.
         * @param {function(Object):*} callback Function to execute, invoked with an element as 
         * argument. To break the iteration you can optionally return false in the callback.
         */
        tree.postorderTraversal = function (callback) {

            function postorderRecursive(node, callback, signal) {
                if (node === undefined || signal.stop) {
                    return;
                }
                postorderRecursive(node.leftCh, callback, signal);
                if (signal.stop) {
                    return;
                }
                postorderRecursive(node.rightCh, callback, signal);
                if (signal.stop) {
                    return;
                }
                signal.stop = callback(node.element) === false;
            }


            postorderRecursive(root, callback, {
                stop: false
            });
        };

        /**
         * Executes the provided function once per element present in the tree in level-order.
         * @param {function(Object):*} callback Function to execute, invoked with an element as 
         * argument. To break the iteration you can optionally return false in the callback.
         */
        tree.levelTraversal = function (callback) {

            function levelAux(node, callback) {
                var queue = buckets.Queue();
                if (node !== undefined) {
                    queue.enqueue(node);
                }
                while (!queue.isEmpty()) {
                    node = queue.dequeue();
                    if (callback(node.element) === false) {
                        return;
                    }
                    if (node.leftCh !== undefined) {
                        queue.enqueue(node.leftCh);
                    }
                    if (node.rightCh !== undefined) {
                        queue.enqueue(node.rightCh);
                    }
                }
            }

            levelAux(root, callback);
        };

        /**
         * Returns the minimum element of the tree.
         * @return {*} The minimum element of the tree or undefined if the tree
         * is empty.
         */
        tree.minimum = function () {
            if (tree.isEmpty()) {
                return undefined;
            }
            return minimumAux(root).element;
        };

        /**
         * Returns the maximum element of the tree.
         * @return {*} The maximum element of the tree or undefined if the tree
         * is empty.
         */
        tree.maximum = function () {

            function maximumAux(node) {
                while (node.rightCh !== undefined) {
                    node = node.rightCh;
                }
                return node;
            }

            if (tree.isEmpty()) {
                return undefined;
            }

            return maximumAux(root).element;
        };

        /**
         * Executes the provided function once per element present in the tree in in-order.
         * Equivalent to inorderTraversal.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked with an element argument. To break the iteration you can
         * optionally return false in the callback.
         */
        tree.forEach = function (callback) {
            tree.inorderTraversal(callback);
        };

        /**
         * Returns an array containing all the elements in the tree in in-order.
         * @return {Array} An array containing all the elements in the tree in in-order.
         */
        tree.toArray = function () {
            var array = [];
            tree.inorderTraversal(function (element) {
                array.push(element);
            });
            return array;
        };

        /**
         * Returns the height of the tree.
         * @return {number} The height of the tree or -1 if it's empty.
         */
        tree.height = function () {

            function heightAux(node) {
                if (node === undefined) {
                    return -1;
                }
                return Math.max(heightAux(node.leftCh), heightAux(node.rightCh)) + 1;
            }

            function heightRecursive(node) {
                if (node === undefined) {
                    return -1;
                }
                return Math.max(heightAux(node.leftCh), heightAux(node.rightCh)) + 1;
            }

            return heightRecursive(root);
        };

        /**
         * Returns true if the tree is equal to another tree.
         * Two trees are equal if they have the same elements.
         * @param {buckets.BSTree} other The other tree.
         * @return {boolean} True if the tree is equal to the given tree.
         */
        tree.equals = function (other) {
            var isEqual;

            if (buckets.isUndefined(other) || typeof other.levelTraversal !== 'function') {
                return false;
            }
            if (tree.size() !== other.size()) {
                return false;
            }

            isEqual = true;
            other.forEach(function (element) {
                isEqual = tree.contains(element);
                return isEqual;
            });
            return isEqual;
        };

        return tree;
    };


    /**
     * Creates an empty dictionary.
     * @class <p>Dictionaries map keys to values, each key can map to at most one value.
     * This implementation accepts any kind of objects as keys.</p>
     *
     * <p>If the keys are custom objects, a function that converts keys to unique
     * strings must be provided at construction time.</p>
     * <p>Example:</p>
     * <pre>
     * function petToString(pet) {
     *  return pet.name;
     * }
     * </pre>
     * @constructor
     * @param {function(Object):string=} toStrFunction Optional function used
     * to convert keys to unique strings. If the keys aren't strings or if toString()
     * is not appropriate, a custom function which receives a key and returns a
     * unique string must be provided.
     */
    buckets.Dictionary = function (toStrFunction) {

        /** 
         * @exports dictionary as buckets.Dictionary
         * @private
         */
        var dictionary = {},
            // Object holding the key-value pairs.
            table = {},
            // Number of keys in the dictionary.
            nElements = 0,
            // Function to convert keys unique to strings.
            toStr = toStrFunction || buckets.defaultToString,
            // Special string to prefix keys and avoid name collisions with existing properties.
            keyPrefix = '/$ ';

        /**
         * Returns the value associated with the specified key in the dictionary.
         * @param {Object} key The key.
         * @return {*} The mapped value or
         * undefined if the dictionary contains no mapping for the provided key.
         */
        dictionary.get = function (key) {
            var pair = table[keyPrefix + toStr(key)];
            if (buckets.isUndefined(pair)) {
                return undefined;
            }
            return pair.value;
        };

        /**
         * Associates the specified value with the specified key in the dictionary.
         * If the dictionary previously contained a mapping for the key, the old
         * value is replaced by the specified value.
         * @param {Object} key The key.
         * @param {Object} value Value to be mapped with the specified key.
         * @return {*} Previous value associated with the provided key, or undefined if
         * there was no mapping for the key or the key/value is undefined.
         */
        dictionary.set = function (key, value) {
            var ret, k, previousElement;
            if (buckets.isUndefined(key) || buckets.isUndefined(value)) {
                return undefined;
            }

            k = keyPrefix + toStr(key);
            previousElement = table[k];
            if (buckets.isUndefined(previousElement)) {
                nElements += 1;
                ret = undefined;
            } else {
                ret = previousElement.value;
            }
            table[k] = {
                key: key,
                value: value
            };
            return ret;
        };

        /**
         * Removes the value associated with the specified key from the dictionary if it exists.
         * @param {Object} key The key.
         * @return {*} Removed value associated with the specified key, or undefined if
         * there was no mapping for the key.
         */
        dictionary.remove = function (key) {
            var k = keyPrefix + toStr(key),
                previousElement = table[k];
            if (!buckets.isUndefined(previousElement)) {
                delete table[k];
                nElements -= 1;
                return previousElement.value;
            }
            return undefined;
        };

        /**
         * Returns an array containing all the keys in the dictionary.
         * @return {Array} An array containing all the keys in the dictionary.
         */
        dictionary.keys = function () {
            var array = [],
                name;
            for (name in table) {
                if (Object.prototype.hasOwnProperty.call(table, name)) {
                    array.push(table[name].key);
                }
            }
            return array;
        };

        /**
         * Returns an array containing all the values in the dictionary.
         * @return {Array} An array containing all the values in the dictionary.
         */
        dictionary.values = function () {
            var array = [],
                name;
            for (name in table) {
                if (Object.prototype.hasOwnProperty.call(table, name)) {
                    array.push(table[name].value);
                }
            }
            return array;
        };

        /**
         * Executes the provided function once per key-value pair
         * present in the dictionary.
         * @param {function(Object,Object):*} callback Function to execute. Receives
         * 2 arguments: key and value. To break the iteration you can
         * optionally return false inside the callback.
         */
        dictionary.forEach = function (callback) {
            var name, pair, ret;
            for (name in table) {
                if (Object.prototype.hasOwnProperty.call(table, name)) {
                    pair = table[name];
                    ret = callback(pair.key, pair.value);
                    if (ret === false) {
                        return;
                    }
                }
            }
        };

        /**
         * Returns true if the dictionary contains a mapping for the specified key.
         * @param {Object} key The key.
         * @return {boolean} True if the dictionary contains a mapping for the
         * specified key.
         */
        dictionary.containsKey = function (key) {
            return !buckets.isUndefined(dictionary.get(key));
        };

        /**
         * Removes all keys and values from the dictionary.
         * @this {buckets.Dictionary}
         */
        dictionary.clear = function () {
            table = {};
            nElements = 0;
        };

        /**
         * Returns the number of key-value pais in the dictionary.
         * @return {number} The number of key-value mappings in the dictionary.
         */
        dictionary.size = function () {
            return nElements;
        };

        /**
         * Returns true if the dictionary contains no keys.
         * @return {boolean} True if this dictionary contains no mappings.
         */
        dictionary.isEmpty = function () {
            return nElements <= 0;
        };

        /**
         * Returns true if the dictionary is equal to another dictionary.
         * Two dictionaries are equal if they have the same key-value pairs.
         * @param {buckets.Dictionary} other The other dictionary.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function to check if two values are equal. If the values in the dictionaries
         * are custom objects you should provide a custom equals function, otherwise
         * the === operator is used to check equality between values.
         * @return {boolean} True if the dictionary is equal to the given dictionary.
         */
        dictionary.equals = function (other, equalsFunction) {
            var eqf, isEqual;
            if (buckets.isUndefined(other) || typeof other.keys !== 'function') {
                return false;
            }
            if (dictionary.size() !== other.size()) {
                return false;
            }
            eqf = equalsFunction || buckets.defaultEquals;
            isEqual = true;
            other.forEach(function (k, v) {
                isEqual = eqf(dictionary.get(k), v);
                return isEqual;
            });
            return isEqual;
        };

        return dictionary;
    };


    /**
     * Creates an empty binary heap.
     * @class
     * <p>A heap is a binary tree that maintains the heap property:
     * Every node is less than or equal to each of its children. 
     * This implementation uses an array as the underlying storage.</p>
     * <p>If the inserted elements are custom objects, a compare function must be provided 
     * at construction time, otherwise the <=, === and >= operators are
     * used to compare elements.</p>
     * <p>Example:</p>
     * <pre>
     * function compare(a, b) {
     *  if (a is less than b by some ordering criterion) {
     *     return -1;
     *  } if (a is greater than b by the ordering criterion) {
     *     return 1;
     *  }
     *  // a must be equal to b
     *  return 0;
     * }
     * </pre>
     *
     * <p>To create a Max-Heap (greater elements on top) you can a provide a
     * reverse compare function.</p>
     * <p>Example:</p>
     *
     * <pre>
     * function reverseCompare(a, b) {
     *  if (a is less than b by some ordering criterion) {
     *     return 1;
     *  } if (a is greater than b by the ordering criterion) {
     *     return -1;
     *  }
     *  // a must be equal to b
     *  return 0;
     * }
     * </pre>
     *
     * @constructor
     * @param {function(Object,Object):number=} compareFunction Optional
     * function used to compare two elements. Must return a negative integer,
     * zero, or a positive integer as the first argument is less than, equal to,
     * or greater than the second.
     */
    buckets.Heap = function (compareFunction) {

        /** 
         * @exports heap as buckets.Heap
         * @private
         */
        var heap = {},
            // Array used to store the elements of the heap.
            data = [],
            // Function used to compare elements.
            compare = compareFunction || buckets.defaultCompare;

        // Moves the node at the given index up to its proper place in the heap.
        function siftUp(index) {
            var parent;
            // Returns the index of the parent of the node at the given index.
            function parentIndex(nodeIndex) {
                return Math.floor((nodeIndex - 1) / 2);
            }

            parent = parentIndex(index);
            while (index > 0 && compare(data[parent], data[index]) > 0) {
                buckets.arrays.swap(data, parent, index);
                index = parent;
                parent = parentIndex(index);
            }
        }

        // Moves the node at the given index down to its proper place in the heap.
        function siftDown(nodeIndex) {
            var min;
            // Returns the index of the left child of the node at the given index.
            function leftChildIndex(nodeIndex) {
                return (2 * nodeIndex) + 1;
            }

            // Returns the index of the right child of the node at the given index.
            function rightChildIndex(nodeIndex) {
                return (2 * nodeIndex) + 2;
            }

            // Returns the index of the smaller child node if it exists, -1 otherwise.
            function minIndex(leftChild, rightChild) {
                if (rightChild >= data.length) {
                    if (leftChild >= data.length) {
                        return -1;
                    }
                    return leftChild;
                }
                if (compare(data[leftChild], data[rightChild]) <= 0) {
                    return leftChild;
                }
                return rightChild;
            }

            // Minimum child index
            min = minIndex(leftChildIndex(nodeIndex), rightChildIndex(nodeIndex));

            while (min >= 0 && compare(data[nodeIndex], data[min]) > 0) {
                buckets.arrays.swap(data, min, nodeIndex);
                nodeIndex = min;
                min = minIndex(leftChildIndex(nodeIndex), rightChildIndex(nodeIndex));
            }
        }

        /**
         * Retrieves but does not remove the root (minimum) element of the heap.
         * @return {*} The value at the root of the heap. Returns undefined if the
         * heap is empty.
         */
        heap.peek = function () {
            if (data.length > 0) {
                return data[0];
            }
            return undefined;
        };

        /**
         * Adds the given element into the heap.
         * @param {*} element The element.
         * @return True if the element was added or false if it is undefined.
         */
        heap.add = function (element) {
            if (buckets.isUndefined(element)) {
                return undefined;
            }
            data.push(element);
            siftUp(data.length - 1);
            return true;
        };

        /**
         * Retrieves and removes the root (minimum) element of the heap.
         * @return {*} The removed element or
         * undefined if the heap is empty.
         */
        heap.removeRoot = function () {
            var obj;
            if (data.length > 0) {
                obj = data[0];
                data[0] = data[data.length - 1];
                data.splice(data.length - 1, 1);
                if (data.length > 0) {
                    siftDown(0);
                }
                return obj;
            }
            return undefined;
        };

        /**
         * Returns true if the heap contains the specified element.
         * @param {Object} element Element to search for.
         * @return {boolean} True if the Heap contains the specified element, false
         * otherwise.
         */
        heap.contains = function (element) {
            var equF = buckets.compareToEquals(compare);
            return buckets.arrays.contains(data, element, equF);
        };

        /**
         * Returns the number of elements in the heap.
         * @return {number} The number of elements in the heap.
         */
        heap.size = function () {
            return data.length;
        };

        /**
         * Checks if the heap is empty.
         * @return {boolean} True if the heap contains no elements; false
         * otherwise.
         */
        heap.isEmpty = function () {
            return data.length <= 0;
        };

        /**
         * Removes all the elements from the heap.
         */
        heap.clear = function () {
            data.length = 0;
        };

        /**
         * Executes the provided function once per element present in the heap in
         * no particular order.
         * @param {function(Object):*} callback Function to execute,
         * invoked with an element as argument. To break the iteration you can
         * optionally return false.
         */
        heap.forEach = function (callback) {
            buckets.arrays.forEach(data, callback);
        };

        /**
         * Returns an array containing all the elements in the heap in no
         * particular order.
         * @return {Array.<*>} An array containing all the elements in the heap
         * in no particular order.
         */
        heap.toArray = function () {
            return buckets.arrays.copy(data);
        };

        /**
         * Returns true if the binary heap is equal to another heap.
         * Two heaps are equal if they have the same elements.
         * @param {buckets.Heap} other The other heap.
         * @return {boolean} True if the heap is equal to the given heap.
         */
        heap.equals = function (other) {
            var thisArray, otherArray, eqF;

            if (buckets.isUndefined(other) || typeof other.removeRoot !== 'function') {
                return false;
            }
            if (heap.size() !== other.size()) {
                return false;
            }

            thisArray = heap.toArray();
            otherArray = other.toArray();
            eqF = buckets.compareToEquals(compare);
            thisArray.sort(compare);
            otherArray.sort(compare);

            return buckets.arrays.equals(thisArray, otherArray, eqF);
        };

        return heap;
    };


    /**
     * Creates an empty Linked List.
     * @class A linked list is a sequence of items arranged one after 
     * another. The size is not fixed and it can grow or shrink 
     * on demand. One of the main benefits of a linked list is that 
     * you can add or remove elements at both ends in constant time. 
     * One disadvantage of a linked list against an array is 
     * that it doesn’t provide constant time random access.
     * @constructor
     */
    buckets.LinkedList = function () {

        /** 
         * @exports list as buckets.LinkedList
         * @private
         */
        var list = {},
            // Number of elements in the list
            nElements = 0,
            // First node in the list
            firstNode,
            // Last node in the list
            lastNode;

        // Returns the node at the specified index.
        function nodeAtIndex(index) {
            var node, i;
            if (index < 0 || index >= nElements) {
                return undefined;
            }
            if (index === (nElements - 1)) {
                return lastNode;
            }
            node = firstNode;
            for (i = 0; i < index; i += 1) {
                node = node.next;
            }
            return node;
        }

        /**
         * Adds an element to the list.
         * @param {Object} item Element to be added.
         * @param {number=} index Optional index to add the element. If no index is specified
         * the element is added to the end of the list.
         * @return {boolean} True if the element was added or false if the index is invalid
         * or if the element is undefined.
         */
        list.add = function (item, index) {
            var newNode, prev;

            if (buckets.isUndefined(index)) {
                index = nElements;
            }
            if (index < 0 || index > nElements || buckets.isUndefined(item)) {
                return false;
            }
            newNode = {
                element: item,
                next: undefined
            };
            if (nElements === 0) {
                // First node in the list.
                firstNode = newNode;
                lastNode = newNode;
            } else if (index === nElements) {
                // Insert at the end.
                lastNode.next = newNode;
                lastNode = newNode;
            } else if (index === 0) {
                // Change first node.
                newNode.next = firstNode;
                firstNode = newNode;
            } else {
                prev = nodeAtIndex(index - 1);
                newNode.next = prev.next;
                prev.next = newNode;
            }
            nElements += 1;
            return true;
        };

        /**
         * Returns the first element in the list.
         * @return {*} The first element in the list or undefined if the list is
         * empty.
         */
        list.first = function () {
            if (firstNode !== undefined) {
                return firstNode.element;
            }
            return undefined;
        };

        /**
         * Returns the last element in the list.
         * @return {*} The last element in the list or undefined if the list is
         * empty.
         */
        list.last = function () {
            if (lastNode !== undefined) {
                return lastNode.element;
            }
            return undefined;
        };

        /**
         * Returns the element at the specified position in the list.
         * @param {number} index Desired index.
         * @return {*} The element at the given index or undefined if the index is
         * out of bounds.
         */
        list.elementAtIndex = function (index) {
            var node = nodeAtIndex(index);
            if (node === undefined) {
                return undefined;
            }
            return node.element;
        };


        /**
         * Returns the index of the first occurrence of the
         * specified element, or -1 if the list does not contain the element.
         * <p>If the elements inside the list are
         * not comparable with the === operator, a custom equals function should be
         * provided to perform searches, that function must receive two arguments and
         * return true if they are equal, false otherwise. Example:</p>
         *
         * <pre>
         * var petsAreEqualByName = function(pet1, pet2) {
         *  return pet1.name === pet2.name;
         * }
         * </pre>
         * @param {Object} item Element to search for.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function used to check if two elements are equal.
         * @return {number} The index in the list of the first occurrence
         * of the specified element, or -1 if the list does not contain the
         * element.
         */
        list.indexOf = function (item, equalsFunction) {
            var equalsF = equalsFunction || buckets.defaultEquals,
                currentNode = firstNode,
                index = 0;
            if (buckets.isUndefined(item)) {
                return -1;
            }

            while (currentNode !== undefined) {
                if (equalsF(currentNode.element, item)) {
                    return index;
                }
                index += 1;
                currentNode = currentNode.next;
            }
            return -1;
        };

        /**
         * Returns true if the list contains the specified element.
         * <p>If the elements inside the list are
         * not comparable with the === operator, a custom equals function should be
         * provided to perform searches, that function must receive two arguments and
         * return true if they are equal, false otherwise. Example:</p>
         *
         * <pre>
         * var petsAreEqualByName = function(pet1, pet2) {
         *  return pet1.name === pet2.name;
         * }
         * </pre>
         * @param {Object} item Element to search for.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function used to check if two elements are equal.
         * @return {boolean} True if the list contains the specified element, false
         * otherwise.
         */
        list.contains = function (item, equalsFunction) {
            return (list.indexOf(item, equalsFunction) >= 0);
        };

        /**
         * Removes the first occurrence of the specified element in the list.
         * <p>If the elements inside the list are
         * not comparable with the === operator, a custom equals function should be
         * provided to perform searches, that function must receive two arguments and
         * return true if they are equal, false otherwise. Example:</p>
         * <pre>
         * var petsAreEqualByName = function(pet1, pet2) {
         *  return pet1.name === pet2.name;
         * }
         * </pre>
         * @param {Object} item Element to be removed from the list, if present.
         * @return {boolean} True if the list contained the specified element.
         */
        list.remove = function (item, equalsFunction) {
            var equalsF = equalsFunction || buckets.defaultEquals,
                currentNode = firstNode,
                previous;

            if (nElements < 1 || buckets.isUndefined(item)) {
                return false;
            }

            while (currentNode !== undefined) {

                if (equalsF(currentNode.element, item)) {

                    if (currentNode === firstNode) {
                        firstNode = firstNode.next;
                        if (currentNode === lastNode) {
                            lastNode = undefined;
                        }
                    } else if (currentNode === lastNode) {
                        lastNode = previous;
                        previous.next = currentNode.next;
                        currentNode.next = undefined;
                    } else {
                        previous.next = currentNode.next;
                        currentNode.next = undefined;
                    }
                    nElements = nElements - 1;
                    return true;
                }
                previous = currentNode;
                currentNode = currentNode.next;
            }
            return false;
        };

        /**
         * Removes all the elements from the list.
         */
        list.clear = function () {
            firstNode = undefined;
            lastNode = undefined;
            nElements = 0;
        };

        /**
         * Returns true if the list is equal to another list.
         * Two lists are equal if they have the same elements in the same order.
         * @param {buckets.LinkedList} other The other list.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function to check if two elements are equal. If the elements in the lists
         * are custom objects you should provide a custom equals function, otherwise
         * the === operator is used to check equality between elements.
         * @return {boolean} true if the list is equal to the given list.
         */
        list.equals = function (other, equalsFunction) {
            var eqf = equalsFunction || buckets.defaultEquals,
                isEqual = true,
                node = firstNode;

            if (buckets.isUndefined(other) || typeof other.elementAtIndex !== 'function') {
                return false;
            }
            if (list.size() !== other.size()) {
                return false;
            }

            other.forEach(function (element) {
                isEqual = eqf(element, node.element);
                node = node.next;
                return isEqual;
            });

            return isEqual;
        };

        /**
         * Removes the element at the specified position in the list.
         * @param {number} index Given index.
         * @return {*} Removed element or undefined if the index is out of bounds.
         */
        list.removeElementAtIndex = function (index) {
            var element, previous;

            if (index < 0 || index >= nElements) {
                return undefined;
            }

            if (nElements === 1) {
                //First node in the list.
                element = firstNode.element;
                firstNode = undefined;
                lastNode = undefined;
            } else {
                previous = nodeAtIndex(index - 1);
                if (previous === undefined) {
                    element = firstNode.element;
                    firstNode = firstNode.next;
                } else if (previous.next === lastNode) {
                    element = lastNode.element;
                    lastNode = previous;
                }
                if (previous !== undefined) {
                    element = previous.next.element;
                    previous.next = previous.next.next;
                }
            }
            nElements -= 1;
            return element;
        };

        /**
         * Executes the provided function once per element present in the list in order.
         * @param {function(Object):*} callback Function to execute, it is
         * invoked with one argument: the element value, to break the iteration you can
         * optionally return false inside the callback.
         */
        list.forEach = function (callback) {
            var currentNode = firstNode;
            while (currentNode !== undefined) {
                if (callback(currentNode.element) === false) {
                    break;
                }
                currentNode = currentNode.next;
            }
        };

        /**
         * Reverses the order of the elements in the linked list (makes the last
         * element first, and the first element last).
         * @memberOf buckets.LinkedList
         */
        list.reverse = function () {
            var current = firstNode,
                previous,
                temp;
            while (current !== undefined) {
                temp = current.next;
                current.next = previous;
                previous = current;
                current = temp;
            }
            temp = firstNode;
            firstNode = lastNode;
            lastNode = temp;
        };


        /**
         * Returns an array containing all the elements in the list in proper
         * sequence.
         * @return {Array.<*>} An array containing all the elements in the list,
         * in proper sequence.
         */
        list.toArray = function () {
            var result = [];
            list.forEach(function (element) {
                result.push(element);
            });
            return result;
        };

        /**
         * Returns the number of elements in the list.
         * @return {number} The number of elements in the list.
         */
        list.size = function () {
            return nElements;
        };

        /**
         * Returns true if the list contains no elements.
         * @return {boolean} true if the list contains no elements.
         */
        list.isEmpty = function () {
            return nElements <= 0;
        };

        return list;
    };


    /**
     * Creates an empty multi dictionary.
     * @class <p>A multi dictionary is a special kind of dictionary that holds
     * multiple values against each key. Setting a value into the dictionary will
     * add the value to a list at that key. Getting a key will return a list
     * holding all the values associated with that key.
     * This implementation accepts any kind of objects as keys.</p>
     *
     * <p>If the keys are custom objects, a function that converts keys to unique strings must be
     * provided at construction time.</p>
     * <p>Example:</p>
     * <pre>
     * function petToString(pet) {
     *  return pet.type + ' ' + pet.name;
     * }
     * </pre>
     * <p>If the values are custom objects, a function to check equality between values
     * must be provided.</p>
     * <p>Example:</p>
     * <pre>
     * function petsAreEqualByAge(pet1,pet2) {
     *  return pet1.age===pet2.age;
     * }
     * </pre>
     * @constructor
     * @param {function(Object):string=} toStrFunction optional function
     * to convert keys to strings. If the keys aren't strings or if toString()
     * is not appropriate, a custom function which receives a key and returns a
     * unique string must be provided.
     * @param {function(Object,Object):boolean=} valuesEqualsFunction optional
     * function to check if two values are equal.
     *
     */
    buckets.MultiDictionary = function (toStrFunction, valuesEqualsFunction) {

        /** 
         * @exports multiDict as buckets.MultiDictionary
         * @private
         */
        var multiDict = {},
            // Call the parent constructor
            parent = new buckets.Dictionary(toStrFunction),
            equalsF = valuesEqualsFunction || buckets.defaultEquals;

        /**
         * Returns an array holding the values associated with
         * the specified key.
         * @param {Object} key The key.
         * @return {Array} An array holding the values or an 
         * empty array if the dictionary contains no 
         * mappings for the provided key.
         */
        multiDict.get = function (key) {
            var values = parent.get(key);
            if (buckets.isUndefined(values)) {
                return [];
            }
            return buckets.arrays.copy(values);
        };

        /**
         * Associates the specified value with the specified key if
         * it's not already present.
         * @param {Object} key The Key.
         * @param {Object} value The value to associate.
         * @return {boolean} True if the value was not already associated with that key.
         */
        multiDict.set = function (key, value) {
            var array;
            if (buckets.isUndefined(key) || buckets.isUndefined(value)) {
                return false;
            }
            if (!multiDict.containsKey(key)) {
                parent.set(key, [value]);
                return true;
            }
            array = parent.get(key);
            if (buckets.arrays.contains(array, value, equalsF)) {
                return false;
            }
            array.push(value);
            return true;
        };

        /**
         * Removes the specified value from the list of values associated with the
         * provided key. If a value isn't given, all values associated with the specified
         * key are removed.
         * @param {Object} key The key.
         * @param {Object=} value Optional argument to specify the element to remove
         * from the list of values associated with the given key.
         * @return {*} True if the dictionary changed, false if the key doesn't exist or
         * if the specified value isn't associated with the given key.
         */
        multiDict.remove = function (key, value) {
            var v, array;
            if (buckets.isUndefined(value)) {
                v = parent.remove(key);
                if (buckets.isUndefined(v)) {
                    return false;
                }
                return true;
            }
            array = parent.get(key);
            if (buckets.arrays.remove(array, value, equalsF)) {
                if (array.length === 0) {
                    parent.remove(key);
                }
                return true;
            }
            return false;
        };

        /**
         * Returns an array containing all the keys in the dictionary.
         * @return {Array} An array containing all the keys in the dictionary.
         */
        multiDict.keys = function () {
            return parent.keys();
        };

        /**
         * Returns an array containing all the values in the dictionary.
         * @return {Array} An array containing all the values in the dictionary.
         */
        multiDict.values = function () {
            var values = parent.values(),
                array = [],
                i,
                j,
                v;
            for (i = 0; i < values.length; i += 1) {
                v = values[i];
                for (j = 0; j < v.length; j += 1) {
                    array.push(v[j]);
                }
            }
            return array;
        };

        /**
         * Returns true if the dictionary has at least one value associatted with the specified key.
         * @param {Object} key The key.
         * @return {boolean} True if the dictionary has at least one value associatted
         * the specified key.
         */
        multiDict.containsKey = function (key) {
            return parent.containsKey(key);
        };

        /**
         * Removes all keys and values from the dictionary.
         */
        multiDict.clear = function () {
            return parent.clear();
        };

        /**
         * Returns the number of keys in the dictionary.
         * @return {number} The number of keys in the dictionary.
         */
        multiDict.size = function () {
            return parent.size();
        };

        /**
         * Returns true if the dictionary contains no mappings.
         * @return {boolean} True if the dictionary contains no mappings.
         */
        multiDict.isEmpty = function () {
            return parent.isEmpty();
        };

        /**
         * Executes the provided function once per key
         * present in the multi dictionary.
         * @param {function(Object, Array):*} callback Function to execute. Receives
         * 2 arguments: key and an array of values. To break the iteration you can
         * optionally return false inside the callback.
         */
        multiDict.forEach = function (callback) {
            return parent.forEach(callback);
        };

        /**
         * Returns true if the multi dictionary is equal to another multi dictionary.
         * Two dictionaries are equal if they have the same keys and the same values per key.
         * @param {buckets.MultiDictionary} other The other dictionary.
         * @return {boolean} True if the dictionary is equal to the given dictionary.
         */
        multiDict.equals = function (other) {
            var isEqual = true,
                thisValues;

            if (buckets.isUndefined(other) || typeof other.values !== 'function') {
                return false;
            }
            if (multiDict.size() !== other.size()) {
                return false;
            }

            other.forEach(function (key, otherValues) {
                thisValues = multiDict.get(key) || [];
                if (thisValues.length !== otherValues.length) {
                    isEqual = false;
                } else {
                    buckets.arrays.forEach(thisValues, function (value) {
                        isEqual = buckets.arrays.contains(otherValues, value, equalsF);
                        return isEqual;
                    });
                }
                return isEqual;
            });
            return isEqual;
        };

        return multiDict;
    };


    /**
     * Creates an empty priority queue.
     * @class <p>In a priority queue each element is associated with a "priority",
     * elements are dequeued in highest-priority-first order (the elements with the
     * highest priority are dequeued first). This implementation uses a binary 
     * heap as the underlying storage.</p>
     *
     * <p>If the inserted elements are custom objects, a compare function must be provided,
     * otherwise the <=, === and >= operators are used to compare object priority.</p>
     * <p>Example:</p>
     * <pre>
     * function compare(a, b) {
     *  if (a is less than b by some ordering criterion) {
     *     return -1;
     *  } if (a is greater than b by the ordering criterion) {
     *     return 1;
     *  }
     *  // a must be equal to b
     *  return 0;
     * }
     * </pre>
     * @constructor
     * @param {function(Object,Object):number=} compareFunction Optional
     * function used to compare two element priorities. Must return a negative integer,
     * zero, or a positive integer as the first argument is less than, equal to,
     * or greater than the second.
     */
    buckets.PriorityQueue = function (compareFunction) {

        /** 
         * @exports pQueue as buckets.PriorityQueue
         * @private
         */
        var pQueue = {},
            // Reversed compare function
            compare = buckets.reverseCompareFunction(compareFunction),
            // Underlying storage
            heap = new buckets.Heap(compare);

        /**
         * Inserts the specified element into the priority queue.
         * @param {Object} element The element to insert.
         * @return {boolean} True if the element was inserted, or false if it's undefined.
         */
        pQueue.enqueue = function (element) {
            return heap.add(element);
        };

        /**
         * Inserts the specified element into the priority queue. It's equivalent to enqueue.
         * @param {Object} element The element to insert.
         * @return {boolean} True if the element was inserted, or false if it's undefined.
         */
        pQueue.add = function (element) {
            return heap.add(element);
        };

        /**
         * Retrieves and removes the highest priority element of the queue.
         * @return {*} The highest priority element of the queue,
         * or undefined if the queue is empty.
         */
        pQueue.dequeue = function () {
            var elem;
            if (heap.size() !== 0) {
                elem = heap.peek();
                heap.removeRoot();
                return elem;
            }
            return undefined;
        };

        /**
         * Retrieves, but does not remove, the highest priority element of the queue.
         * @return {*} The highest priority element of the queue, or undefined if the queue is empty.
         */
        pQueue.peek = function () {
            return heap.peek();
        };

        /**
         * Returns true if the priority queue contains the specified element.
         * @param {Object} element Element to search for.
         * @return {boolean} True if the priority queue contains the specified element,
         * false otherwise.
         */
        pQueue.contains = function (element) {
            return heap.contains(element);
        };

        /**
         * Checks if the priority queue is empty.
         * @return {boolean} True if and only if the priority queue contains no items, false
         * otherwise.
         */
        pQueue.isEmpty = function () {
            return heap.isEmpty();
        };

        /**
         * Returns the number of elements in the priority queue.
         * @return {number} The number of elements in the priority queue.
         */
        pQueue.size = function () {
            return heap.size();
        };

        /**
         * Removes all elements from the priority queue.
         */
        pQueue.clear = function () {
            heap.clear();
        };

        /**
         * Executes the provided function once per element present in the queue in
         * no particular order.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked one element as argument. To break the iteration you can
         * optionally return false inside the callback.
         */
        pQueue.forEach = function (callback) {
            heap.forEach(callback);
        };

        /**
         * Returns an array containing all the elements in the queue in no
         * particular order.
         * @return {Array.<*>} An array containing all the elements in the queue
         * in no particular order.
         */
        pQueue.toArray = function () {
            return heap.toArray();
        };

        /**
         * Returns true if the queue is equal to another queue.
         * Two priority queues are equal if they have the same elements.
         * @param {buckets.PriorityQueue} other The other queue.
         * @return {boolean} True if the queue is equal to the given queue.
         */
        pQueue.equals = function (other) {
            var thisArray, otherArray, eqF;

            if (buckets.isUndefined(other) || typeof other.dequeue !== 'function') {
                return false;
            }
            if (pQueue.size() !== other.size()) {
                return false;
            }

            thisArray = pQueue.toArray();
            otherArray = other.toArray();
            eqF = buckets.compareToEquals(compare);
            thisArray.sort(compare);
            otherArray.sort(compare);

            return buckets.arrays.equals(thisArray, otherArray, eqF);
        };

        return pQueue;
    };


    /**
     * Creates an empty queue.
     * @class A queue is a First-In-First-Out (FIFO) data structure, the first
     * element added to the queue will be the first one to be removed. This
     * implementation uses a linked list as the underlying storage.
     * @constructor
     */
    buckets.Queue = function () {

        /** 
         * @exports queue as buckets.Queue
         * @private
         */
        var queue = {},
            // Underlying list containing the elements.
            list = new buckets.LinkedList();

        /**
         * Inserts the specified element into the end of the queue.
         * @param {Object} elem The element to insert.
         * @return {boolean} True if the element was inserted, or false if it's undefined.
         */
        queue.enqueue = function (elem) {
            return list.add(elem);
        };

        /**
         * Inserts the specified element into the end of the queue. Equivalent to enqueue.
         * @param {Object} elem The element to insert.
         * @return {boolean} True if the element was inserted, or false if it's undefined.
         */
        queue.add = function (elem) {
            return list.add(elem);
        };

        /**
         * Retrieves and removes the head of the queue.
         * @return {*} The head of the queue, or undefined if the queue is empty.
         */
        queue.dequeue = function () {
            var elem;
            if (list.size() !== 0) {
                elem = list.first();
                list.removeElementAtIndex(0);
                return elem;
            }
            return undefined;
        };

        /**
         * Retrieves, but does not remove, the head of the queue.
         * @return {*} The head of the queue, or undefined if the queue is empty.
         */
        queue.peek = function () {
            if (list.size() !== 0) {
                return list.first();
            }
            return undefined;
        };

        /**
         * Returns the number of elements in the queue.
         * @return {number} The number of elements in the queue.
         */
        queue.size = function () {
            return list.size();
        };

        /**
         * Returns true if the queue contains the specified element.
         * <p>If the elements inside the queue are
         * not comparable with the === operator, a custom equals function should be
         * provided to perform searches, the function must receive two arguments and
         * return true if they are equal, false otherwise. Example:</p>
         *
         * <pre>
         * var petsAreEqualByName = function(pet1, pet2) {
         *  return pet1.name === pet2.name;
         * }
         * </pre>
         * @param {Object} elem Element to search for.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function to check if two elements are equal.
         * @return {boolean} True if the queue contains the specified element,
         * false otherwise.
         */
        queue.contains = function (elem, equalsFunction) {
            return list.contains(elem, equalsFunction);
        };

        /**
         * Checks if the queue is empty.
         * @return {boolean} True if and only if the queue contains no items.
         */
        queue.isEmpty = function () {
            return list.size() <= 0;
        };

        /**
         * Removes all the elements from the queue.
         */
        queue.clear = function () {
            list.clear();
        };

        /**
         * Executes the provided function once per each element present in the queue in
         * FIFO order.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked an element as argument, to break the iteration you can
         * optionally return false inside the callback.
         */
        queue.forEach = function (callback) {
            list.forEach(callback);
        };

        /**
         * Returns an array containing all the elements in the queue in FIFO
         * order.
         * @return {Array.<*>} An array containing all the elements in the queue
         * in FIFO order.
         */
        queue.toArray = function () {
            return list.toArray();
        };

        /**
         * Returns true if the queue is equal to another queue.
         * Two queues are equal if they have the same elements in the same order.
         * @param {buckets.Queue} other The other queue.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function to check if two elements are equal. If the elements in the queues
         * are custom objects you should provide a custom equals function, otherwise
         * the === operator is used to check equality between elements.
         * @return {boolean} True if the queue is equal to the given queue.
         */
        queue.equals = function (other, equalsFunction) {
            var eqf, isEqual, thisElement;
            if (buckets.isUndefined(other) || typeof other.dequeue !== 'function') {
                return false;
            }
            if (queue.size() !== other.size()) {
                return false;
            }
            eqf = equalsFunction || buckets.defaultEquals;
            isEqual = true;
            other.forEach(function (element) {
                thisElement = queue.dequeue();
                queue.enqueue(thisElement);
                isEqual = eqf(thisElement, element);
                return isEqual;
            });
            return isEqual;
        };

        return queue;
    };


    /**
     * Creates an empty set.
     * @class <p>A set is a data structure that contains no duplicate items.</p>
     * <p>If the inserted elements are custom objects, a function
     * that converts elements to unique strings must be provided at construction time. 
     * <p>Example:</p>
     * <pre>
     * function petToString(pet) {
     *  return pet.type + ' ' + pet.name;
     * }
     * </pre>
     *
     * @param {function(Object):string=} toStringFunction Optional function used
     * to convert elements to unique strings. If the elements aren't strings or if toString()
     * is not appropriate, a custom function which receives an object and returns a
     * unique string must be provided.
     */
    buckets.Set = function (toStringFunction) {

        /** 
         * @exports theSet as buckets.Set
         * @private
         */
        var theSet = {},
            // Underlying storage.
            dictionary = new buckets.Dictionary(toStringFunction);

        /**
         * Returns true if the set contains the specified element.
         * @param {Object} element Element to search for.
         * @return {boolean} True if the set contains the specified element,
         * false otherwise.
         */
        theSet.contains = function (element) {
            return dictionary.containsKey(element);
        };

        /**
         * Adds the specified element to the set if it's not already present.
         * @param {Object} element The element to insert.
         * @return {boolean} True if the set did not already contain the specified element.
         */
        theSet.add = function (element) {
            if (theSet.contains(element) || buckets.isUndefined(element)) {
                return false;
            }
            dictionary.set(element, element);
            return true;
        };

        /**
         * Performs an intersection between this and another set.
         * Removes all values that are not present in this set and the given set.
         * @param {buckets.Set} otherSet Other set.
         */
        theSet.intersection = function (otherSet) {
            theSet.forEach(function (element) {
                if (!otherSet.contains(element)) {
                    theSet.remove(element);
                }
            });
        };

        /**
         * Performs a union between this and another set.
         * Adds all values from the given set to this set.
         * @param {buckets.Set} otherSet Other set.
         */
        theSet.union = function (otherSet) {
            otherSet.forEach(function (element) {
                theSet.add(element);
            });
        };

        /**
         * Performs a difference between this and another set.
         * Removes all the values that are present in the given set from this set.
         * @param {buckets.Set} otherSet other set.
         */
        theSet.difference = function (otherSet) {
            otherSet.forEach(function (element) {
                theSet.remove(element);
            });
        };

        /**
         * Checks whether the given set contains all the elements of this set.
         * @param {buckets.Set} otherSet Other set.
         * @return {boolean} True if this set is a subset of the given set.
         */
        theSet.isSubsetOf = function (otherSet) {
            var isSub = true;

            if (theSet.size() > otherSet.size()) {
                return false;
            }

            theSet.forEach(function (element) {
                if (!otherSet.contains(element)) {
                    isSub = false;
                    return false;
                }
            });
            return isSub;
        };

        /**
         * Removes the specified element from the set.
         * @return {boolean} True if the set contained the specified element, false
         * otherwise.
         */
        theSet.remove = function (element) {
            if (!theSet.contains(element)) {
                return false;
            }
            dictionary.remove(element);
            return true;
        };

        /**
         * Executes the provided function once per element
         * present in the set.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked an element as argument. To break the iteration you can
         * optionally return false inside the callback.
         */
        theSet.forEach = function (callback) {
            dictionary.forEach(function (k, v) {
                return callback(v);
            });
        };

        /**
         * Returns an array containing all the elements in the set in no particular order.
         * @return {Array} An array containing all the elements in the set.
         */
        theSet.toArray = function () {
            return dictionary.values();
        };

        /**
         * Returns true if the set contains no elements.
         * @return {boolean} True if the set contains no elements.
         */
        theSet.isEmpty = function () {
            return dictionary.isEmpty();
        };

        /**
         * Returns the number of elements in the set.
         * @return {number} The number of elements in the set.
         */
        theSet.size = function () {
            return dictionary.size();
        };

        /**
         * Removes all the elements from the set.
         */
        theSet.clear = function () {
            dictionary.clear();
        };

        /**
         * Returns true if the set is equal to another set.
         * Two sets are equal if they have the same elements.
         * @param {buckets.Set} other The other set.
         * @return {boolean} True if the set is equal to the given set.
         */
        theSet.equals = function (other) {
            var isEqual;
            if (buckets.isUndefined(other) || typeof other.isSubsetOf !== 'function') {
                return false;
            }
            if (theSet.size() !== other.size()) {
                return false;
            }

            isEqual = true;
            other.forEach(function (element) {
                isEqual = theSet.contains(element);
                return isEqual;
            });
            return isEqual;
        };

        return theSet;
    };


    /**
     * Creates an empty Stack.
     * @class A Stack is a Last-In-First-Out (LIFO) data structure, the last
     * element added to the stack will be the first one to be removed. This
     * implementation uses a linked list as the underlying storage.
     * @constructor
     */
    buckets.Stack = function () {

        /** 
         * @exports stack as buckets.Stack
         * @private
         */
        var stack = {},
            // Underlying list containing the elements.
            list = new buckets.LinkedList();

        /**
         * Pushes an element onto the top of the stack.
         * @param {Object} elem The element.
         * @return {boolean} True if the element was pushed or false if it's undefined.
         */
        stack.push = function (elem) {
            return list.add(elem, 0);
        };

        /**
         * Pushes an element onto the top of the stack. Equivalent to push.
         * @param {Object} elem The element.
         * @return {boolean} true If the element was pushed or false if it's undefined.
         */
        stack.add = function (elem) {
            return list.add(elem, 0);
        };

        /**
         * Removes the element at the top of the stack and returns it.
         * @return {*} The element at the top of the stack or undefined if the
         * stack is empty.
         */
        stack.pop = function () {
            return list.removeElementAtIndex(0);
        };

        /**
         * Returns the element at the top of the stack without removing it.
         * @return {*} The element at the top of the stack or undefined if the
         * stack is empty.
         */
        stack.peek = function () {
            return list.first();
        };

        /**
         * Returns the number of elements in the stack.
         * @return {number} The number of elements in the stack.
         */
        stack.size = function () {
            return list.size();
        };

        /**
         * Returns true if the stack contains the specified element.
         * <p>If the elements inside the stack are
         * not comparable with the === operator, a custom equals function must be
         * provided to perform searches, that function must receive two arguments and
         * return true if they are equal, false otherwise. Example:</p>
         *
         * <pre>
         * var petsAreEqualByName = function(pet1, pet2) {
         *  return pet1.name === pet2.name;
         * }
         * </pre>
         * @param {Object} elem Element to search for.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function used to check if two elements are equal.
         * @return {boolean} True if the stack contains the specified element,
         * false otherwise.
         */
        stack.contains = function (elem, equalsFunction) {
            return list.contains(elem, equalsFunction);
        };

        /**
         * Checks if the stack is empty.
         * @return {boolean} True if and only if this stack contains no elements, false
         * otherwise.
         */
        stack.isEmpty = function () {
            return list.isEmpty();
        };

        /**
         * Removes all the elements from the stack.
         */
        stack.clear = function () {
            list.clear();
        };

        /**
         * Executes the provided function once per element present in the stack in
         * LIFO order.
         * @param {function(Object):*} callback Function to execute, it's
         * invoked with an element as argument. To break the iteration you can
         * optionally return false inside the callback.
         */
        stack.forEach = function (callback) {
            list.forEach(callback);
        };

        /**
         * Returns an array containing all the elements in the stack in LIFO
         * order.
         * @return {Array.<*>} An array containing all the elements in the stack
         * in LIFO order.
         */
        stack.toArray = function () {
            return list.toArray();
        };

        /**
         * Returns true if the stack is equal to another stack.
         * Two stacks are equal if they have the same elements in the same order.
         * @param {buckets.Stack} other The other stack.
         * @param {function(Object,Object):boolean=} equalsFunction Optional
         * function to check if two elements are equal. If the elements in the stacks
         * are custom objects you should provide a custom equals function, otherwise
         * the === operator is used to check equality between elements.
         * @return {boolean} True if the stack is equal to the given stack.
         */
        stack.equals = function (other, equalsFunction) {
            var eqf, isEqual, thisElement;
            if (buckets.isUndefined(other) || typeof other.peek !== 'function') {
                return false;
            }
            if (stack.size() !== other.size()) {
                return false;
            }

            eqf = equalsFunction || buckets.defaultEquals;
            isEqual = true;
            other.forEach(function (element) {
                thisElement = stack.pop();
                list.add(thisElement);
                isEqual = eqf(thisElement, element);
                return isEqual;
            });

            return isEqual;
        };

        return stack;
    };


    return buckets;

}));

},{}]},{},[2]);

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes:
  // - FullyConn is fully connected dot products
  // - ConvLayer does convolutions (so weight sharing spatially)
  // putting them together in one file because they are very similar

  // Encapsulates DropConnect configuration, layers, etc.
  var DropConnect = function(opt) {
    var opt = opt || {};
    this.keep_probability = opt.keep_probability;
    this.num_gaussian_samples = opt.num_gaussian_samples;
    this.activation_layer = global.build_layer({
      // Unused, as we just cal forward()
      in_sx: 1,
      in_sy: 1,
      in_depth: 1,
      type: opt.activation
    });
  };

  DropConnect.prototype = {
    fromJSON: function(json) {
      this.keep_probability = json.keep_probability;
      this.num_gaussian_samples = json.num_gaussian_samples;
      this.activation_layer = global.build_layer({
        // Unused, as we just cal forward()
        in_sx: 1,
        in_sy: 1,
        in_depth: 1,
        type: json.activation
      });
    },
    toJSON: function() {
      var opt = {};
      opt.keep_probability = this.keep_probability;
      opt.num_gaussian_samples = this.num_gaussian_samples;
      opt.activation_type = this.activation_layer.layer_type;
    }
  };

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
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            // convolve centered at this particular location
            // could be bit more efficient, going for correctness first
            var a = 0.0;
            for(var fx=0;fx<f.sx;fx++) {
              for(var fy=0;fy<f.sy;fy++) {
                for(var fd=0;fd<f.depth;fd++) {
                  var oy = y+fy; // coordinates in the original input array coordinates
                  var ox = x+fx;
                  if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                    //a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
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
      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {
            // convolve and add up the gradients. 
            // could be more efficient, going for correctness first
            var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            for(var fx=0;fx<f.sx;fx++) {
              for(var fy=0;fy<f.sy;fy++) {
                for(var fd=0;fd<f.depth;fd++) {
                  var oy = y+fy;
                  var ox = x+fx;
                  if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                    // forward prop calculated: a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
                    //f.add_grad(fx, fy, fd, V.get(ox, oy, fd) * chain_grad);
                    //V.add_grad(ox, oy, fd, f.get(fx, fy, fd) * chain_grad);

                    // avoid function call overhead and use Vols directly for efficiency
                    var ix1 = ((V.sx * oy)+ox)*V.depth+fd;
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
      json.drop_connect = this.drop_connect.toJSON();
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
    this.drop_connect = typeof opt.drop_connect !== 'undefined' ? DropConnect(opt.drop_connect) : null;
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
  };

  FullyConnLayer.prototype = {
    dropConnectEnabled: function() {
      return this.drop_connect !== null;
    },

    initializeDropConnectMask: function() {
      this.drop_connect_masks = [];
      var keep_probability = this.dropConnectEnabled() ? this.drop_connect.keep_probability : 1.0;
      for(var i = 0; i < this.filters.length; i++) {
        this.drop_connect_masks[i] =
          global.bernoulliMask(1, 1, this.num_inputs, keep_probability);
      }
    },

    forward: function(V, is_training) {
      if (is_training) {
        return this.forwardTrain(V);
      } else {
        return this.forwardPredict(V);
      }
    },

    forwardTrain: function(V) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      this.initializeDropConnectMask();
      var Vw = V.w;
      for(var i=0;i<this.out_depth;i++) {
        var a = 0.0;
        var wi = this.filters[i].w;
        var weight_mask = this.drop_connect_masks[i].weight_mask.w;
        for(var d=0;d<this.num_inputs;d++) {
          a += Vw[d] * wi[d] * weight_mask[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i] * this.drop_connect_masks[i].bias_mask;
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    },

    // Algorithm 2 in Regularization of Neural Networks using
    // DropConnect - http://cs.nyu.edu/~wanli/dropc/dropc.pdf
    averageGaussianActivation: function(Vw, wi) {
      var mu = 0.0;
      var sigma_squared = 0.0;
      for(var d = 0; d < this.out_depth; d++) {
        mu += Vw[d] * wi[d];
        sigma_squared += (Vw[d] * Vw[d]) * (wi[d] * wi[d]);
      }
      var scaled_mu = this.drop_connect_prop * mu;
      var scaled_sigma = Math.sqrt(this.drop_connect_prop * sigma_squared);
      var sample_units = new Vol(1, 1, this.drop_connect.num_gaussian_samples, 0.0);
      for (var d = 0; d < sample_units.w.length; d++) {
        sample_units.w[d] = global.randn(scaled_mu, scaled_sigma);
      }

      var output_activations = this.drop_connect.activation_layer.forward(sample_units);
      var sum_output = 0.0;
      for (var d = 0; d < output_activations.w.length; d++) {
        sum_output += output_activations.w[d];
      }
      return sum_output / output_activations.w.length;
    },

    forwardPredict: function(V) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      var Vw = V.w;
      for(var i=0;i<this.out_depth;i++) {
        var wi = this.filters[i].w;
        var a = 0.0;
        if (this.dropConnectEnabled()) {
          a = this.averageGaussianActivation(Vw, wi);
        } else {
          // Simply compute the \sum_j W_ij v_i
          for(var d=0;d<this.num_inputs;d++) {
            a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
          }
          a += this.biases.w[i];
        }
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
        var weight_mask = this.drop_connect_masks[i].weight_mask.w;
        for(var d=0;d<this.num_inputs;d++) {
          V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
          tfi.dw[d] += V.w[d]*chain_grad * weight_mask[d]; // grad wrt params
        }
        this.biases.dw[i] += chain_grad * this.drop_connect_masks[i].bias_mask;
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({
          params: this.filters[i].w,
          grads: this.filters[i].dw,
          l1_decay_mul: this.l1_decay_mul,
          l2_decay_mul: this.l2_decay_mul
        });
      }
      response.push({
        params: this.biases.w,
        grads: this.biases.dw,
        l1_decay_mul: 0.0,
        l2_decay_mul: 0.0
      });
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

      // DropConnect
      if (this.dropConnectEnabled()) {
        json.drop_connect = this.drop_connect.toJSON();
      }
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

      // DropConnect
      if (typeof json.drop_connect !== 'undefined') {
        this.drop_connect = new DropConnect();
        this.drop_connect.fromJSON(json.drop_connect);
      }

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

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
          
          if(def.type==='softmax' || def.type==='svm') {
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
            if(typeof def.activation === 'undefined') {
              continue;
            }
            if(typeof def.drop_connect_keep_prop !== 'undefined') {
              def.drop_connect = {};
              def.drop_connect.keep_probability = def.drop_connect_keep_prob;
              def.drop_connect.num_gaussian_samples = def.drop_connect_num_gaussian_samples;
              def.drop_connect.activation = def.activation;
            }
            if(def.activation === 'relu') {
              def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
              // otherwise it's technically possible that a relu unit will never turn on (by chance)
              // and will never get any gradient and never contribute any computation. Dead relu.
            }
          }

          if(typeof def.tensor !== 'undefined') {
            // apply quadratic transform so that the upcoming multiply will include
            // quadratic terms, equivalent to doing a tensor product
            if(def.tensor) {
              new_defs.push({type: 'quadtransform'});
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
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

        this.layers.push(global.build_layer(def));
      }
    },

    // forward prop the network. A trainer will pass in is_training = true
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
        if(t==='tanh') { L = new global.TanhLayer(); }
        if(t==='dropout') { L = new global.DropoutLayer(); }
        if(t==='conv') { L = new global.ConvLayer(); }
        if(t==='pool') { L = new global.PoolLayer(); }
        if(t==='lrn') { L = new global.LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new global.SoftmaxLayer(); }
        if(t==='regression') { L = new global.RegressionLayer(); }
        if(t==='fc') { L = new global.FullyConnLayer(); }
        if(t==='maxout') { L = new global.MaxoutLayer(); }
        if(t==='quadtransform') { L = new global.QuadTransformLayer(); }
        if(t==='svm') { L = new global.SVMLayer(); }
        L.fromJSON(Lj);
        this.layers.push(L);
      }
    }
  }

  var build_layer = function(def) {
    switch(def.type) {
    case 'fc': return new global.FullyConnLayer(def);
    case 'lrn': return new global.LocalResponseNormalizationLayer(def);
    case 'dropout': return new global.DropoutLayer(def);
    case 'input': return new global.InputLayer(def);
    case 'softmax': return new global.SoftmaxLayer(def);
    case 'regression': return new global.RegressionLayer(def);
    case 'conv': return new global.ConvLayer(def);
    case 'pool': return new global.PoolLayer(def);
    case 'relu': return new global.ReluLayer(def);
    case 'sigmoid': return new global.SigmoidLayer(def);
    case 'tanh': return new global.TanhLayer(def);
    case 'maxout': return new global.MaxoutLayer(def);
    case 'quadtransform': return new global.QuadTransformLayer(def);
    case 'svm': return new global.SVMLayer(def);
    default: console.log('ERROR: UNRECOGNIZED LAYER TYPE!');
    }
  };

  global.Net = Net;
  global.build_layer = build_layer;
})(convnetjs);

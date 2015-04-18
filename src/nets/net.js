import Vol from "./convnet_vol.js";
import assert from "./convnet_utils.js";

export default class Net {

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  constructor(options){
    this.layers = [];
  }

  // takes a list of layer definitions and creates the network layer objects
  makeLayers(defs) {

    // few checks
    assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
    assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

    // desugar layer_defs for adding activation, dropout layers etc
    var desugar = function() {
      var new_defs = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        
        if(def.type==='softmax' || def.type==='svm') {
          // add an fc layer here, there is no reason the user should
          // have to worry about this and we almost always want to
          new_defs.push({type:'fc', num_neurons: def.num_classes});
        } else if(def.type==='regression') {
          // add an fc layer here, there is no reason the user should
          // have to worry about this and we almost always want to
          new_defs.push({type:'fc', num_neurons: def.num_neurons});
        } else if((def.type==='fc' || def.type==='conv') && typeof(def.bias_pref) === 'undefined'){
          def.bias_pref = 0.0;
          if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
            def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
            // otherwise it's technically possible that a relu unit will never turn on (by chance)
            // and will never get any gradient and never contribute any computation. Dead relu.
          }
        }

        new_defs.push(def);

        if(typeof def.activation !== 'undefined') {
          if(def.activation==='relu') { 
            new_defs.push({type:'relu'}); 
          } else if (def.activation==='sigmoid') { 
            new_defs.push({type:'sigmoid'}); 
          } else if (def.activation==='tanh') { 
            new_defs.push({type:'tanh'}); 
          } else if (def.activation==='maxout') {
            // create maxout activation, and pass along group size, if provided
            var gs = def.group_size !== 'undefined' ? def.group_size : 2;
            new_defs.push({type:'maxout', group_size:gs});
          } else { 
            console.log('ERROR unsupported activation ' + def.activation); 
          }
        }
        if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
          new_defs.push({type:'dropout', drop_prob: def.drop_prob});
        }

      }
      return new_defs;
    }

    defs = desugar(defs);

    // create the layers
    this.layers = defs.mapPar((x, i) => {

      if(i > 0){
        var prev = this.layers[i-1];
        def.in_sx = prev.out_sx;
        def.in_sy = prev.out_sy;
        def.in_depth = prev.out_depth;
      }

      switch(def.type) {
        case 'fc': 
          return new global.FullyConnLayer(def); 
          break;
        case 'lrn': 
          return new global.LocalResponseNormalizationLayer(def); 
          break;
        case 'dropout': 
          return new global.DropoutLayer(def); 
          break;
        case 'input': 
          return new global.InputLayer(def); 
          break;
        case 'softmax': 
          return new global.SoftmaxLayer(def); 
          break;
        case 'regression': 
          return new global.RegressionLayer(def); 
          break;
        case 'conv': 
          return new global.ConvLayer(def); 
          break;
        case 'pool': 
          return new global.PoolLayer(def); 
          break;
        case 'relu': 
          return new global.ReluLayer(def); 
          break;
        case 'sigmoid': 
          return new global.SigmoidLayer(def); 
          break;
        case 'tanh': 
          return new global.TanhLayer(def); 
          break;
        case 'maxout': 
          return new global.MaxoutLayer(def); 
          break;
        case 'svm': 
          return new global.SVMLayer(def)); 
          break;
        default: 
          console.error('ERROR: Unrecognised layer type: ' + def.type);
      }

    });
  }

  // forward prop the network. 
  // The trainer class passes is_training = true, but when this function is
  // called from outside (not from the trainer), it defaults to prediction mode
  forward(V, is_training) {
    if(typeof(is_training) === 'undefined') is_training = false;
    var act = this.layers[0].forward(V, is_training);
    for(var i=1;i<this.layers.length;i++) {
      act = this.layers[i].forward(act, is_training);
    }
    return act;
  }

  getCostLoss(V, y) {
    this.forward(V, false);
    var N = this.layers.length;
    var loss = this.layers[N-1].backward(y);
    return loss;
  }
  
  // backprop: compute gradients wrt all parameters
  backward(y) {
    var N = this.layers.length;
    var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
    for(var i=N-2;i>=0;i--) { // first layer assumed input
      this.layers[i].backward();
    }
    return loss;
  }

  getParamsAndGrads() {
    // accumulate parameters and gradients for the entire network
    var response = [];
    for(var i=0;i<this.layers.length;i++) {
      var layer_reponse = this.layers[i].getParamsAndGrads();
      for(var j=0;j<layer_reponse.length;j++) {
        response.push(layer_reponse[j]);
      }
    }
    return response;
  }

  getPrediction() {
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
  }

  toJSON() {
    var json = {};
    json.layers = [];
    for(var i=0;i<this.layers.length;i++) {
      json.layers.push(this.layers[i].toJSON());
    }
    return json;
  }

  fromJSON(json) {
    this.layers = json.mapPar(x => {
      var t = x.layer_type;
      var L;
      switch()
      if(t==='input') { L = new InputLayer(); }
      if(t==='relu') { L = new ReluLayer(); }
      if(t==='sigmoid') { L = new SigmoidLayer(); }
      if(t==='tanh') { L = new TanhLayer(); }
      if(t==='dropout') { L = new DropoutLayer(); }
      if(t==='conv') { L = new ConvLayer(); }
      if(t==='pool') { L = new PoolLayer(); }
      if(t==='lrn') { L = new LocalResponseNormalizationLayer(); }
      if(t==='softmax') { L = new SoftmaxLayer(); }
      if(t==='regression') { L = new RegressionLayer(); }
      if(t==='fc') { L = new FullyConnLayer(); }
      if(t==='maxout') { L = new MaxoutLayer(); }
      if(t==='svm') { L = new SVMLayer(); }
      L.fromJSON(x);
      return L;
    });
  }  
}
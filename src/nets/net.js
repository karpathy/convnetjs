import {
  ConvLayer, 
  DropoutLayer, 
  FullyConnLayer, 
  InputLayer, 
  MaxoutLayer, 
  PoolLayer, 
  RegressionLayer, 
  ReluLayer, 
  SigmoidLayer, 
  SoftmaxLayer, 
  SVMLayer, 
  TanhLayer
} from "../layers/index.js";

export default class Net {

  layers = []

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  constructor(defs = [], options = {}){

    // few checks
    if(defs.length < 2){
      throw new Error('Error! At least one input layer and one loss layer are required.');
    } else if (defs[0].type !== 'input' && defs[0].constructor !== InputLayer) {
      throw new Error('Error! First layer must be the input layer, to declare size of inputs');
    }

    let in_sx, in_sy, in_depth;

    for(let i = 0; i < defs.length; i++){
      let def = defs[i];

      if((def.type === 'softmax' || def.constructor.name === 'SoftmaxLayer' ||
          def.type === 'svm' || def.constructor.name === 'SVMLayer') && 
        (this.layers[this.layers.length-1].num_neurons != def.num_classes)){
        this.layers.push(new FullyConnLayer({
          num_neurons: def.num_classes,
          in_sx : in_sx,
          in_sy : in_sy,
          in_depth : in_depth
        }));
      } else if((def.type === 'regression' || def.constructor.name === 'RegressionLayer') && 
        (this.layers[this.layers.length-1].num_neurons != def.num_neurons)){
        this.layers.push(new FullyConnLayer({
          num_neurons: def.num_neurons,
          in_sx : in_sx,
          in_sy : in_sy,
          in_depth : in_depth
        }));
      } else if((def.type === 'fc' || def.type === 'conv' || 
        def.constructor.name === 'FullyConnLayer' || def.constructor.name === 'ReluLayer') && 
        def.bias_pref == undefined){
        def.bias_pref = (def.activation === 'relu') ? 0.1 : 0.0; // relus like a bit of positive bias to get gradients early
        // otherwise it's technically possible that a relu unit will never turn on (by chance)
        // and will never get any gradient and never contribute any computation. Dead relu.
      } 

      if(def.constructor.name === 'Object'){
        switch(def.type) {
          case 'fc': 
            this.layers.push(new FullyConnLayer(def)); 
            break;
          case 'lrn': 
            this.layers.push(new LocalResponseNormalizationLayer(def)); 
            break;
          case 'dropout': 
            this.layers.push(new DropoutLayer(def)); 
            break;
          case 'input': 
            this.layers.push(new InputLayer(def)); 
            break;
          case 'softmax': 
            this.layers.push(new SoftmaxLayer(def)); 
            break;
          case 'regression': 
            this.layers.push(new RegressionLayer(def)); 
            break;
          case 'conv': 
            this.layers.push(new ConvLayer(def)); 
            break;
          case 'pool': 
            this.layers.push(new PoolLayer(def)); 
            break;
          case 'relu': 
            this.layers.push(new ReluLayer(def)); 
            break;
          case 'sigmoid': 
            this.layers.push(new SigmoidLayer(def)); 
            break;
          case 'tanh': 
            this.layers.push(new TanhLayer(def)); 
            break;
          case 'maxout': 
            this.layers.push(new MaxoutLayer(def)); 
            break;
          case 'svm': 
            this.layers.push(new SVMLayer(def)); 
            break;
          default:
            throw new Error("Unrecognised layer type: " + def.type);
        }
      }else{
        this.layers.push(def);
      }

      in_sx = def.out_sx; in_sy = def.out_sy; in_depth = def.out_depth;

      if (def.type === 'con' || def.type === 'fc'){
        if(def.activation === 'relu' && (defs[i+1].constructor.name !== 'ReluLayer' || defs[i+1].layer_type !== 'relu')){
              this.layers.push(new ReluLayer({
                in_sx : in_sx,
                in_sy : in_sy,
                in_depth : in_depth
              }));
              in_sx = this.layers[this.layer.length-1].out_sx; 
              in_sy = this.layers[this.layer.length-1].out_sy; 
              in_depth = this.layers[this.layer.length-1].out_depth;
            }else if(def.activation === 'sigmoid' && (defs[i+1].constructor.name !== 'SigmoidLayer' || defs[i+1].layer_type !== 'sigmoid')){
              this.layers.push(new SigmoidLayer({
                in_sx : in_sx,
                in_sy : in_sy,
                in_depth : in_depth
              }));
              in_sx = this.layers[this.layer.length-1].out_sx; 
              in_sy = this.layers[this.layer.length-1].out_sy; 
              in_depth = this.layers[this.layer.length-1].out_depth;
            }else if(def.activation === 'tanh' && (defs[i+1].constructor.name !== 'TanhLayer' || defs[i+1].layer_type !== 'tanh')){
              this.layers.push(new TanhLayer({
                in_sx : in_sx,
                in_sy : in_sy,
                in_depth : in_depth
              }));
              in_sx = this.layers[this.layer.length-1].out_sx; 
              in_sy = this.layers[this.layer.length-1].out_sy; 
              in_depth = this.layers[this.layer.length-1].out_depth;
            }else if(def.activation === 'maxout' && (defs[i+1].constructor.name !== 'MaxoutLayer' || defs[i+1].layer_type !== 'maxout')){
              this.layers.push(new MaxoutLayer({
                in_sx : in_sx,
                in_sy : in_sy,
                in_depth : in_depth,
                group_size : def.group_size || 2
              }));
              in_sx = this.layers[this.layer.length-1].out_sx; 
              in_sy = this.layers[this.layer.length-1].out_sy; 
              in_depth = this.layers[this.layer.length-1].out_depth;
            }else {
              throw new Error("Unsupported activation: " + def.activation);
            }
          }

      if(def.drop_prob != undefined && (def.type !== 'dropout' || def.constructor.name !== 'DropoutLayer')){
        this.layers.push(new DropoutLayer({
          drop_prob : def.drop_prob,
          in_sx : in_sx,
          in_sy : in_sy,
          in_depth : in_depth
        }));
      }
    }
  }

  // forward prop the network. 
  // The trainer class passes is_training = true, but when this function is
  // called from outside (not from the trainer), it defaults to prediction mode
  forward(V, use_webgl = false, is_training = false) {
    var act = this.layers[0].forward(V, use_webgl, is_training);
    for(var i = 1; i < this.layers.length; i++) {
      act = this.layers[i].forward(act, use_webgl, is_training);
    }
    return act;
  }

  getCostLoss(V, y) {
    this.forward(V, false);
    return this.layers[this.layers.length-1].backward(y);
  }
  
  // backprop: compute gradients wrt all parameters
  backward(y, use_webgl = false, is_training = false) {
    var N = this.layers.length;
    var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
    for(var i = N - 2; i >= 0; i--) { // first layer assumed input
      this.layers[i].backward();
    }
    return loss;
  }

  getParamsAndGrads() {
    // accumulate parameters and gradients for the entire network
    return [].concat.apply([], [for (layer of this.layers) layer.getParamsAndGrads()]);
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
      if(p[i] > maxv) { 
        maxv = p[i]; 
        maxi = i;
      }
    }
    return maxi; // return index of the class with highest class probability
  }

  toJSON() {
    return this.layers.map(x => x.toJSON());
  }

}

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new Net(json.layers.map(x => {
    switch(x.layer_type){
      case 'input':
        return InputLayer.fromJSON(x); 
      case 'relu':
        return ReluLayer.fromJSON(x); 
      case 'sigmoid':
        return SigmoidLayer.fromJSON(x); 
      case 'tanh':
        return TanhLayer.fromJSON(x); 
      case 'dropout':
        return DropoutLayer.fromJSON(x); 
      case 'conv':
        return ConvLayer.fromJSON(x); 
      case 'pool':
        return PoolLayer.fromJSON(x); 
      case 'lrn':
        return LocalResponseNormalizationLayer.fromJSON(); 
      case 'softmax':
        return SoftmaxLayer.fromJSON(); 
      case 'regression':
        return RegressionLayer.fromJSON(); 
      case 'fc':
        return FullyConnLayer.fromJSON(); 
      case 'maxout':
        return MaxoutLayer.fromJSON(); 
      case 'svm':
        return SVMLayer.fromJSON();
    }
  }));
}  
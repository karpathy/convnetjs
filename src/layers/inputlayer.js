import * as Layer from "./layer.js";

export default class InputLayer extends Layer {

  constructor(opt = {}){
    super(opt);
    // required: depth
    this.out_depth = opt.depth || opt.out_depth || 0;

    // optional: default these dimensions to 1
    this.out_sx = opt.width || opt.sx || opt.out_sx || 1;
    this.out_sy = opt.height || opt.sy || opt.out_sy || 1;
    
    // computed
    this.layer_type = 'input';
  }

  forward(V, is_training) {
    this.in_act = V;
    this.out_act = V;
    return this.out_act; // simply identity function for now
  }

  backward() { 
  }

  getParamsAndGrads() {
    return new Float64Array(0);
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type
    };
  }
}

export function fromJSON(json){
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new InputLayer(json);
}
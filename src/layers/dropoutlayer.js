import * as Layer from "./layer.js";

export default class DropoutLayer extends Layer {

  constructor(opt = {}){
    super();
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'dropout';
    this.drop_prob = typeof opt.drop_prob !== 'undefined' ? opt.drop_prob : 0.5;
    this.dropped = new Float64Array(this.out_sx*this.out_sy*this.out_depth);
  }

  forward(V, is_training = false) {
    this.in_act = V;
    let V2 = V.clone();
    const N = V.w.length;
    if(is_training) {
      // do dropout
      for(var i=0;i<N;i++) {
        if(Math.random()<this.drop_prob) { 
          V2.w[i]=0; this.dropped[i] = true; // drop! 
        } else {
          this.dropped[i] = false;
        }
      }
    } else {
      // scale the activations during prediction
      for(var i=0;i<N;i++) { 
        V2.w[i]*=this.drop_prob; 
      }
    }
    this.out_act = V2;
    return this.out_act; // dummy identity function for now
  }

  backward() {
    let V = this.in_act; // we need to set dw of this
    let chain_grad = this.out_act;
    const N = V.w.length;
    V.dw = new Float64Array(N); // zero out gradient wrt data
    for(var i=0;i<N;i++) {
      if(!(this.dropped[i])) { 
        V.dw[i] = chain_grad.dw[i]; // copy over the gradient
      }
    }
  }

  getParamsAndGrads() {
    return new Float64Array(0);
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      drop_prob : this.drop_prob
    };
  }

}

export function fromJSON(json){
  if(typeof json === "string"){
    json = JSON.parse(json);
  }
  return new DropoutLayer(json);
}
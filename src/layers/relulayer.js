import * as Layer from "./layer.js";

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)

export class ReluLayer extends Layer {

  constructor(opts = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'relu';
  }

  forward(V, is_training = false) {
    this.in_act = V;
    this.out_act = new V.constructor(V);

    let v = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.w).buffer);

    let zeroes = SIMD.float64x2.zero();
    
    // Beautifully succinct, isn't it?
    for(var i = 0; i < v.length; i += 2){
      SIMD.float64x2.store(v2, i, SIMD.float64x2.max(SIMD.float64x2.load(v, i), zeroes));
    }

    return this.out_act;
  }

  backward() {
    let v = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.dw).buffer);

    let zeroes = SIMD.float64x2.zero();
    
    for(var i = 0; i < v.length; i += 2){
      SIMD.float64x2.store(v2, i, SIMD.float64x2.max(SIMD.float64x2.load(v, i), zeroes));
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
      layer_type : this.layer_type
    }
  }

}

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new ReluLayer({
    out_depth : json.out_depth,
    out_sx : json.out_depth,
    out_sy : json.out_sy
  });
}
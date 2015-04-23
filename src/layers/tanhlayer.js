import * as Layer from "./layer.js"

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x) 
// so the output is between -1 and 1.

export class TanhLayer extends Layer{

  constructor(opt = {}){
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'tanh';
  }

  forward(V, is_training) {
    this.in_act = V;
    this.out_act = new V.constructor();

    let v = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.w).buffer);

    let len = (v.length|0)
    
    for(let i = 0; i < len; i += 2){
      SIMD.float64x2.store(v2, i, SIMD.float64x2(Math.tanh(v[i]), Math.tanh(v[i+1])));
    }

    return this.out_act;
  }

  backward() {

    let v = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.dw).buffer);

    let len = (v.length|0);
    let ones = SIMD.float64x2.splat(1.0);

    for(let i = 0; i < len; i += 2){
      let out = SIMD.float64x2.load(v2, i);
      SIMD.float64x2.store(v, i, SIMD.float64x2.mul(SIMD.float64x2.sub(ones, SIMD.float64x2.mul(out, out)), out));
    }

  }

  getParamsAndGrads() {
    return [];
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

export function fromJSON(json) {
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new TanhLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
  });
}
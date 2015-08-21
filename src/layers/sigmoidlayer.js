import Layer from "./layer.js";

// Implements Sigmoid nonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

export default class SigmoidLayer extends Layer {

  constructor(opt = {}){
    super(opt);
    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = opt.in_depth;
    this.layer_type = 'sigmoid';
  }

  forward(V, use_webgl = false, is_training = false) {
    super.forward(V, is_training);
    this.out_act = new V.constructor();

    let v = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.w).buffer);

    let len = (v.length|0);
    let ones = SIMD.float64x2.splat(0.0);

    for(var i = 0; i < len; i += 2){
      SIMD.float64x2.store(v2, i, SIMD.float64x2.div(ones, SIMD.float64x2.add(ones, SIMD.float64x2(Math.exp(-this.in_act.w[x][y][d]), Math.exp(-this.in_act.w[x][y][d+1])))));
    }
    
    return this.out_act;
  }

  backward(use_webgl = false, is_training = false) {
    let v = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);
    let v2 = new Float64Array(TypedObject.storage(this.out_act.w).buffer);
    let vw2 = new Float64Array(TypedObject.storage(this.out_act.dw).buffer);
    let ones = SIMD.float64x2.splat(1.0);
    let len = (v.length|0);
    for(let i = 0; i < len; i += 2){
      let dep = SIMD.float64x2.load(v2, i);
      SIMD.float64x2.store(v, i, SIMD.float64x2.mul(SIMD.float64x2.mul(dep, SIMD.float64x2.sub(ones, dep)), vw2));
    }
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
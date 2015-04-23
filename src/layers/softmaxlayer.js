import * as Layer from "./layer.js";
import * as VolType from "../structures/vol.js";

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)

export class SoftmaxLayer extends Layer {

  constructor(opt = {}){
    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_depth = this.num_inputs;
    this.out_sx = 1;
    this.out_sy = 1;
    this.out_type = new VolType(1, 1, this.out_depth);
    this.layer_type = 'softmax';
    this.es = new Float64Array(this.out_depth);
  }

  forward(V, is_training = false) {
    this.in_act = V;
    this.out_act = new this.out_type();

    let v = new Float64Array(TypedObject.storage(this.in_act.w).buffer);
    let a = new Float64Array(TypedObject.storage(this.out_act.w).buffer);

    // compute max activation
    let len = (v.length | 0)
    let amax = SIMD.float64x2.zero();

    for(let i = 0; i < len; i += 2){
      amax = SIMD.float64x2.max(amax, SIMD.float64x2.load(v, i));
    }

    // compute exponentials (carefully to not blow up)
    let esum = SIMD.float64x2.zero();
    let max = Math.max(Math.max(amax.x, amax.y), Math.max(amax.z, amax.w));

    for(var i = 0; i < this.out_depth; i += 2) {
      var e = SIMD.float64x2(Math.exp(v[i] - max), Math.exp(v[i+1] - max));
      esum = SIMD.float64x2.add(esum, e);
      SIMD.float64x2.store(this.es, i, e);
    }

    // normalize and output to sum to one
    esum = SIMD.float64x2.splat(esum.x + esum.y + esum.z + esum.w);

    for(var i = 0; i < this.out_depth; i += 2) {
      let esi = SIMD.float64x2.div(SIMD.float32x4.load(es, i), esum);
      SIMD.float64x2.store(this.es, i, esi);
      SIMD.float64x2.store(a, i, esi)
    }

    return this.out_act;
  }

  backward(y) {

    // compute and accumulate gradient wrt weights and bias of this layer
    let x = new Float64Array(TypedObject.storage(this.in_act.dw).buffer);

    for(var i = 0; i < this.out_depth; i += 2) {
      SIMD.float64x2.store(x, i, SIMD.float64x4(-(i === y ? 1.0 : 0.0 - this.es[i]), -(i+1 === y ? 1.0 : 0.0 - this.es[i+1])))
    }

    // loss is the class negative log likelihood
    return -Math.log(this.es[y]);
  }

  getParamsAndGrads() { 
    return [];
  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      num_inputs : this.num_inputs
    };
  }

}

export function fromJSON(json) {
  if(typeof json === string){
    json = JSON.parse(json);
  }
  return new SoftmaxLayer({
    out_depth : json.out_depth,
    out_sx : json.out_sx,
    out_sy : json.out_sy,
    num_inputs : json.num_inputs
  });
}
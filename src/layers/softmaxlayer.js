import * as Layer from "./layer.js";
import * as VolType from "../structres/vol.js";

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
  }

  forward(V, is_training) {
    this.in_act = V;

    let A = new this.out_type();

    // compute max activation
    let {sx, sy, depth} = V;
    let amax = SIMD.float32x4.splat(0.0);
    for(let x = 0, i = 0; x < sx, i < this.out_depth; x++, i++){
      for(let y = 0; y < sy, i < this.out_depth; y++, i++){
        for(let d = 0; d < depth, i < this.out_depth; d++, i++){
          amax = SIMD.float32x4.max(amax, SIMD.float32x4(V.w[x][y][d], V.w[x][y][d+1], V.w[x][y][d+2], V.w[x][y][d+3]))
        }
      } 
    }
    let max = Math.max(Math.max(amax.x, amax.y), Math.max(amax.z, amax.w));

    // compute exponentials (carefully to not blow up)
    let es = new Float32Array(this.out_depth);
    let esum = SIMD.float32x4.splat(0.0);

    for(var i=0;i<this.out_depth;i++) {
      var e = SIMD.float32x4(Math.exp(as[i] - max), Math.exp(as[i+1] - max), Math.exp(as[i+2] - max), Math.exp(as[i+3] - max));
      esum = SIMD.float32x4.add(esum, e);
      es[i] = e.x; es[i+1] = e.y; es[i+2] = e.z; es[i+3] = e.w; 
    }

    let sum = (esum.x + esum.y + esum.z + esum.w); 

    // SIMDifying this part probably wouldn't bring any benefits.
    // normalize and output to sum to one
    for(var i = 0; i < this.out_depth; i++) {
      es[i] /= esum;
      A.w[0][0][i] = es[i];
    }

    this.es = es; // save these for backprop
    this.out_act = A;
    return this.out_act;
  }

  backward(y) {

    // compute and accumulate gradient wrt weights and bias of this layer
    var x = this.in_act;
    x.dw = new Float64Array(x.w.length); // zero out the gradient of input Vol

    for(var i=0;i<this.out_depth;i++) {
      var indicator = i === y ? 1.0 : 0.0;
      var mul = -(indicator - this.es[i]);
      x.dw[i] = mul;
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
      json.out_sx : this.out_sx,
      json.out_sy : this.out_sy,
      json.layer_type : this.layer_type,
      json.num_inputs : this.num_inputs
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
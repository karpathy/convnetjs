import VolType from "../structures/vol.js";
import Layer from "./layer.js";

function * fillArray(wif, until){
  for (let i = 0; i < until; i++){
    if(typeof wif === 'function'){
      yield wif();
    } else {
      yield wif;
    }
  }
}

export default class FullyConnLayer extends Layer {

  constructor(opt = {}){
    super(opt);

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = opt.sy || this.sx;
    this.stride = opt.stride || 1; // stride at which we apply filters to input volume
    this.pad = opt.pad || 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = opt.l1_decay_mul || 0.0;
    this.l2_decay_mul = opt.l2_decay_mul || 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    // initializations
    let bias = opt.bias_pref || 0.0;
    const bias_type = new VolType(1, 1, this.out_depth);
    this.bias_type = bias_type;
    this.biases = new bias_type({w:[[[...fillArray(bias, this.out_depth)]]]});

    this.filter_type = new VolType(this.sx, this.sy, this.in_depth);
    this.filters = new (this.filter_type.array(this.out_depth))();
    for(let i = 0; i < this.out_depth; i++) { 
      this.filters[i] = new this.filter_type(); 
    }
  }

  forward(V, is_training = false) {
    super.forward(V, is_training);
    this.out_act = new this.bias_type();
    let ow = new Float64Array(storage(this.out_act.w).buffer);
    let vw = new Float64Array(storage(this.in_act.w).buffer);
    for(var i = 0; i < this.out_depth; i++) {
      let a = SIMD.float64x2.zero();
      let wi = new Float64Array(storage(this.filters[i].w).buffer);
      for(var d = 0; d < this.num_inputs; d += 2) {
        // for efficiency use Vols directly for now
        a = SIMD.float64x2.add(a, SIMD.float64x2.mul(SIMD.float64x2.load(vw, d), SIMD.float64x2.load(wi, d)));
      }
      ow[i] = (a.x + a.y + a.z + a.w) + this.biases.w[0][0][i];
    }
    return this.out_act;
  }

  backward(is_training = false) {
   
    let vd = new Float64Array(storage(this.in_act.dw).buffer);
    let vw = new Float64Array(storage(this.in_act.w).buffer);

    for(let i = 0; i < vd.length; i++){
      vd[i] = 0;
    }

    // compute gradient wrt weights and data
    for(var i = 0; i < this.out_depth; i++) {
      let tfiw = new Float64Array(storage(this.filters[i].w).buffer);
      let tfid = new Float64Array(storage(this.filters[i].dw).buffer);
      let chain_grad = SIMD.float64x2.splat(this.out_act.dw[0][0][i]);
      for(var d = 0; d < this.num_inputs; d += 2) {
        // grad wrt input data
        SIMD.float64x2.store(vd, d, SIMD.float64x2.add(SIMD.float64x2.load(vd, d), SIMD.float64x2.mul(SIMD.float64x2.load(tfiw, d), chain_grad)));
        // grad wrt params
        SIMD.float64x2.store(tfid, d, SIMD.float64x2.add(SIMD.float64x2.load(tfid, d), SIMD.float64x2.mul(SIMD.float64x2.load(vw, d), chain_grad)));
      }
      this.biases.dw[i] += this.out_act.dw[0][0][i];
    }
    
  }

  getParamsAndGrads() {
    var response = new Array(this.out_depth + 1);
    for(var i = 0; i < this.out_depth; i++) {
      response[i] = {
        params: this.filters[i].w, 
        grads: this.filters[i].dw,
        l1_decay_mul: this.l1_decay_mul, 
        l2_decay_mul: this.l2_decay_mul
      };
    }
    response[this.out_depth] = {
      params: this.biases.w, 
      grads: this.biases.dw, 
      l1_decay_mul: 0.0, 
      l2_decay_mul: 0.0
    };
    return response;
  }

  toJSON(){
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      num_inputs : this.num_inputs,
      l1_decay_mul : this.l1_decay_mul,
      l2_decay_mul : this.l2_decay_mul,
      filters : this.filters.map(x => x.toJSON()),
      biases : this.biases.toJSON()
    };
  }

}
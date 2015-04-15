import Vol from "./convnet_vol.js";

export default class FullyConnLayer {

  constructor(opt = {}){

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);

  }

  forward(V, is_training) {
    this.in_act = V;
    var A = new Vol(1, 1, this.out_depth, 0.0);
    var Vw = V.w;
    for(var i=0;i<this.out_depth;i++) {
      var a = 0.0;
      var wi = this.filters[i].w;
      for(var d=0;d<this.num_inputs;d++) {
        a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
      }
      a += this.biases.w[i];
      A.w[i] = a;
    }
    this.out_act = A;
    return this.out_act;
  }

  backward() {
    var V = this.in_act;
    V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
    
    // compute gradient wrt weights and data
    for(var i=0;i<this.out_depth;i++) {
      var tfi = this.filters[i];
      var chain_grad = this.out_act.dw[i];
      for(var d=0;d<this.num_inputs;d++) {
        V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
        tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
      }
      this.biases.dw[i] += chain_grad;
    }
  }

  getParamsAndGrads() {
    var response = [];
    for(var i=0;i<this.out_depth;i++) {
      response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
    }
    response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
    return response;
  }

  toJSON(){
    var json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.num_inputs = this.num_inputs;
    json.l1_decay_mul = this.l1_decay_mul;
    json.l2_decay_mul = this.l2_decay_mul;
    json.filters = this.filters.mapPar(x => x.toJSON());
    json.biases = this.biases.toJSON();
    return json;
  }

  fromJSON(json){
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
    this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
    this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
    this.filters = json.filters.mapPar(x => Vol.fromJSON(x));
    this.biases = Vol.fromJSON(json.biases);
  }

}

export function fromJSON(json){
  if(typeof json === 'string'){
    json = JSON.parse(json);
  }
  return new FullyConnLayer(json);
}
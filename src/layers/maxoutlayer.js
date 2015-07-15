import Layer from "./layer.js";

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size

export class MaxoutLayer extends Layer {

  constructor(opt = {}){
    super(opt);
    // required
    this.group_size = opt.group_size || 2;

    // computed
    this.out_sx = opt.in_sx;
    this.out_sy = opt.in_sy;
    this.out_depth = Math.floor(opt.in_depth / this.group_size);
    this.layer_type = 'maxout';

    this.switches = new Float64Array(this.out_sx*this.out_sy*this.out_depth); // useful for backprop
  }

  forward(V, use_webgl = false, is_training = false) {
    super.forward(V, is_training);
    this.out_act = new V.constructor();

    let v = new Float64Array(storage(this.in_act.w).buffer);
    let v2 = new Float64Array(storage(this.out_act.w).buffer);
    
    let len = (v.length|0)

    for(let i = 0; i < len; i++){
      let ix = i * this.group_size; // base index offset
      let a = v[ix];
      let ai = 0;
      for(var j = 1; j < this.group_size; j++){
        let a2 = v[ix+j];
        if(a2 > a){
          a = a2;
          ai = j;
        }
      }
      v2[i] = a;
      this.switches[i] = ix + ai;
    }

    return this.out_act;
  }

  backward() {

    let v = new Float64Array(storage(this.in_act).buffer);
    let v2 = new Float64Array(storage(this.out_act).buffer);

    let len = (v.length|0);

    // pass the gradient through the appropriate switch
    for(let i = 0; i < len; i++){
      v[i] = 0;
      v[this.switches[i]] = v2[i];
    }

  }

  toJSON() {
    return {
      out_depth : this.out_depth,
      out_sx : this.out_sx,
      out_sy : this.out_sy,
      layer_type : this.layer_type,
      group_size : this.group_size
    };
  }

}
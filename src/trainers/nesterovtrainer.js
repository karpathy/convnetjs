import * as Trainer from "./trainer.js";

export default class NesterovTrainer extends Trainer {
	
	constructor(opts = {}){
		super(opts);
	}

	train(x, y){

		let start_1 = new Date().getTime();
    this.net.forward(x, true); // also set the flag that lets the net know we're just training
    let fwd_time = (new Date().getTime()) - start_1;

    let start_2 = new Date().getTime();
    let cost_loss = this.net.backward(y);
    let bwd_time = (new Date().getTime()) - start_2;

    let l2_decay_loss = SIMD.float32x4.splat(0.0);
    let l1_decay_loss = SIMD.float32x4.splat(0.0);
    let mom = SIMD.float32x4.splat(this.momentum);
    let momm = SIMD.float32x4.splat(1.0 + this.momentum)
    let lr = SIMD.float32x4.splat(this.learning_rate);
      
    this.k++;
    if(this.k % this.batch_size === 0) {

      let pglist = this.net.getParamsAndGrads();

      // initialize lists for accumulators. Will only be done once on first iteration
      if(this.gsum.length === 0) {
        for(let i = 0; i < pglist.length; i++) {
          this.gsum.push(new Float64Array(pglist[i].params.length));
        }
      }

      // perform an update for all sets of weights
      for(let i = 0; i < pglist.length; i++) {

        let {p, g, l2_decay_mul, l1_decay_mul} = pglist[i]; // param, gradient, other options in future (custom learning rate etc)

        // learning rate for some parameters.
        let l2_decay = SIMD.float32x4.splat(this.l2_decay * (l2_decay_mul || 1.0));
        let l1_decay = SIMD.float32x4.splat(this.l1_decay * (l1_decay_mul || 1.0));

        let gsumi = this.gsum[i];
        let xsumi = this.xsum[i];

        let plen = (p.length|0); 

        /*
         * SIMD spaghetti code ahead - best served warm and w/ bolognaise sauce.
         */

        for(let j = 0; j < plen; j += 4) {
          
          let pj = SIMD.float32x4.load(p, j);
          let gj = SIMD.float32x4.load(g, j);
          let gsumij = SIMD.float32x4.load(gsumi, j);

          // accumulate weight decay loss
          l2_decay_loss = SIMD.float32x4.add(l2_decay_loss, SIMD.float32x4.div(SIMD.float32x4.mul(l1_decay, SIMD.float32x4.mul(pj, pj)), SIMD.float32x4.splat(2)));
          l1_decay_loss = SIMD.float32x4.add(l1_decay_loss, SIMD.mul(l1_decay, SIMD.float32x4.abs(pj)));

          let l1grad = SIMD.float32x4.mul(l1_decay, SIMD.float32x4.greaterThan(pj, SIMD.float32x4.splat(0.0)));
          let l2grad = SIMD.float32x4.mul(l2_decay, pj)

          let gij = SIMD.float32x4.div(SIMD.float32x4.add(l2grad, SIMD.float32x4.add(l1grad, gj)), SIMD.float32x4.splat(this.batch_size)); // raw batch gradient
          let dx = gsumij;

          gsumij = SIMD.float32x4.add(SIMD.float32x4.mul(gsumij, mom), SIMD.float32x4.mul(lr, gij));

          SIMD.float32x4.store(gsumi, j, gsumij);

          SIMD.float32x4.store(p, j, SIMD.float32x4.add(SIMD.float32x4.sub(SIMD.float32x4.mul(mom, dx), SIMD.float32x4.mul(momm, gsumij)), pj));          

          // zero out gradient so that we can begin accumulating anew
          SIMD.float32x4.store(g, j, SIMD.float32x4.zero());
        }
      }
    }

    // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
    // in future, TODO: have to completely redo the way loss is done around the network as currently 
    // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
    // and it should all be computed correctly and automatically. 
    return {
      fwd_time: fwd_time, 
      bwd_time: bwd_time,
      l2_decay_loss: (l2_decay_loss.x, l2_decay_loss.y, l2_decay_loss.z, l2_decay_loss.w), 
      l1_decay_loss: (l1_decay_loss.x, l1_decay_loss.y, l1_decay_loss.z, l1_decay_loss.w),
      cost_loss: cost_loss, 
      softmax_loss: cost_loss, 
      loss: cost_loss + (l1_decay_loss.x, l1_decay_loss.y, l1_decay_loss.z, l1_decay_loss.w) + (l2_decay_loss.x, l2_decay_loss.y, l2_decay_loss.z, l2_decay_loss.w)
    }

	}

}
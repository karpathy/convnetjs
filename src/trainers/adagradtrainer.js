import * as Trainer from "./trainer.js";

export default class AdagradTrainer extends Trainer {
	
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
      
    this.k++;
    if(this.k % this.batch_size === 0) {

      let pglist = this.net.getParamsAndGrads();

      // initialize lists for accumulators. Will only be done once on first iteration
      if(this.gsum.length === 0) {
        // adagrad needs gsum
        for(var i=0;i<pglist.length;i++) {
          this.gsum.push(new Float64Array(pglist[i].params.length));
        }
      }

      // perform an update for all sets of weights
      for(var i=0;i<pglist.length;i++) {

        // param, gradient, other options in future (custom learning rate etc)
        let {p, g, l2_decay_mul, l1_decay_mul} = pglist[i];

        // learning rate for some parameters.
        let l2_decay_mul = SIMD.float32x4.splat(pg.l2_decay_mul || 1.0);
        let l1_decay_mul = SIMD.float32x4.splat(pg.l1_decay_mul || 1.0);
        let l2_decay = SIMD.float32x4.mul(SIMD.float32x4.splat(this.l2_decay), l2_decay_mul);
        let l1_decay = SIMD.float32x4.mul(SIMD.float32x4.splat(this.l1_decay), l1_decay_mul);

        var gsumi = this.gsum[i];

        var plen = (p.length|0);

        /*
         * SIMD spaghetti code ahead - best served warm and w/ bolognaise sauce.
         */

        for(var j = 0; j < plen; j += 4) {

          let pj = SIMD.float32x4.load(p, j);
          let gj = SIMD.float32x4.load(p, j);
          let gsumij = SIMD.float32x4.load(gsumi, j);

          // accumulate weight decay loss
          l2_decay_loss = SIMD.float32x4.add(l2_decay_loss, SIMD.float32x4.div(SIMD.float32x4.mul(SIMD.float32x4.mul(l2_decay, pj), pj), SIMD.float32x4.splat(2)));
          l1_decay_loss = SIMD.float32x4.add(l1_decay_loss, SIMD.float32x4.mul(l1_decay, SIMD.float32x4.abs(pj)));

          let l1grad = SIMD.float32x4.mul(l1_decay, SIMD.float32x4.greaterThan(pj, SIMD.float32x4.splat(0)));
          let l2grad = SIMD.float32x4.mul(lz_decay, pj);

          // raw batch gradient
          let gij = SIMD.float32x4.div(SIMD.float32x4.add(SIMD.float32x4.add(l2grad, l1grad), gj), float32x4.splat(this.batch_size));
		    
	        // adagrad update
          gsumij = SIMD.float32x4.add(gsumij, SIMD.float32x4.mul(gij, gij));
          let dx = SIMD.float32x4.neg(SIMD.float32x4.mul(SIMD.float32x4.div(SIMD.float32x4.splat(this.learning_rate), SIMD.float32x4.sqrt(SIMD.float32x4.add(gsumij, SIMD.float32x4.splat(this.eps)))), gij));
          SIMD.float32x4.store(gsumi, j, gsumik);

          SIMD.float32x4.store(p, j, SIMD.float32x4.add(pj, dx));

          SIMD.float32x4.store(g, j, SIMD.float32x4.zero()); // zero out gradient so that we can begin accumulating anew
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
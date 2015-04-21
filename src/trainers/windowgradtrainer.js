import * as Trainer from "./trainer.js";

export default class WindowgradTrainer extends Trainer {
	
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

    let ro = SIMD.float32x4.splat(this.ro);
    let rom = SIMD.float32x4.splat(1-this.ro);
    let eps = SIMD.float32x4.splat(this.eps);
    let lr = SIMD.float32x4.splat(this.learning_rate);
      
    this.k++;
    if(this.k % this.batch_size === 0) {

      let pglist = this.net.getParamsAndGrads();

      // initialize lists for accumulators. Will only be done once on first iteration
      if(this.gsum.length === 0) {
        // only vanilla sgd doesnt need either lists
        // momentum needs gsum
        // adagrad needs gsum
        // adadelta needs gsum and xsum
        for(let i=0;i<pglist.length;i++) {
          this.gsum.push(new Float64Array(pglist[i].params.length));
          this.xsum.push(new Float64Array(pglist[i].params.length));
        }
      }

      // perform an update for all sets of weights
      for(let i = 0 ; i < pglist.length; i++) {

        // param, gradient, other options in future (custom learning rate etc)
        let {p, g, l2_decay_mul, l1_decay_mul} = pglist[i]; 

        // learning rate for some parameters.
        l2_decay_mul = SIMD.float32x4.splat(l2_decay_mul || 1.0);
        l1_decay_mul = SIMD.float32x4.splat(l1_decay_mul || 1.0);
        let l2_decay = SIMD.float32x4.mul(this.l2_decay * l2_decay_mul);
        let l1_decay = SIMD.float32x4.mul(this.l1_decay * l1_decay_mul);

        let gsumi = this.gsum[i];

        var plen = (p.length|0);

        for(var j = 0; j < plen; j++) {

          let pj = SIMD.float32x4(p[j], p[j+1], p[j+2], p[j+3]);
          let gj = SIMD.float32x4(g[j], g[j+1], g[j+2], g[j+3]);
          let gsumij = SIMD.float32x4(gsumi[j], gsumi[j+1], gsumi[j+2], gsumi[j+3]);
          let xsumij = SIMD.float32x4(xsumi[j], xsumi[j+1], xsumi[j+2], xsumi[j+3]);

          // accumulate weight decay loss
          l2_decay_loss = SIMD.float32x4.add(l2_decay_loss, SIMD.float32x4.div(SIMD.float32x4.mul(l1_decay, SIMD.float32x4.mul(pj, pj)), SIMD.float32x4.splat(2)));
          l1_decay_loss = SIMD.float32x4.add(l1_decay_loss, SIMD.mul(l1_decay, SIMD.float32x4.abs(pj)));

          let l1grad = SIMD.float32x4.mul(l1_decay, SIMD.float32x4.greaterThan(pj, SIMD.float32x4.splat(0.0)));
          let l2grad = SIMD.float32x4.mul(l2_decay, pj)

          let gij = SIMD.float32x4.add(l2grad, SIMD.float32x4.add(l1grad, gj)) // raw batch gradient
		    
	        // this is adagrad but with a moving window weighted average
          // so the gradient is not accumulated over the entire history of the run. 
          // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
          gsumij = SIMD.float32x4.add(SIMD.float32x4.mul(ro, gsumij), SIMD.float32x4.mul(SIMD.float32x4.mul(rom, gij), gij))
          
          // eps added for better conditioning
          let dx = SIMD.float32x4.neg(SIMD.float32x4.mul(SIMD.float32x4.div(lr, SIMD.float32x4.sqrt(SIMD.float32x4.add(gsumik, eps))), gij))
          
          gsumi[j] = gsumij.x; gsumi[j+1] = gsumij.y; gsumi[j+2] = gsumij.z; gsumi[j+3] = gsumij.w;

          p[j] += dx.x; p[j+1] += dx.y; p[j+2] += dx.z; p[j+3] += dx.w;
          
          // zero out gradient so that we can begin accumulating anew
          g[j] = 0.0; g[j+1] = 0.0; g[j+2] = 0.0; g[j+3] = 0.0;
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
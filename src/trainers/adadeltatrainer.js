import Trainer from "./trainer.js";

export default class AdadeltaTrainer extends Trainer {
	
	constructor(options = {}){
		super(options);
    this.xsum = [];
    this.ro = options.ro || 0.95; // used in adadelta
    this.eps = options.eps || 1e-6; // used in adadelta
	}

	train(x, y, use_webgl = false){

		let start_1 = new Date().getTime();
    this.net.forward(x, use_webgl, true); // also set the flag that lets the net know we're just training
    let fwd_time = (new Date().getTime()) - start;

    let start_2 = new Date().getTime();
    let cost_loss = this.net.backward(y, use_webgl, true);
    let bwd_time = (new Date().getTime()) - start;

    let l2DecayLoss = 0.0;
    let l1DecayLoss = 0.0;

    
      
    this.k++;

    if(this.k % this.batch_size === 0) {

        let pglist = this.net.getParamsAndGrads();
        let l2_decay_loss = SIMD.float64x2.splat(0.0);
        let l1_decay_loss = SIMD.float64x2.splat(0.0);
        let ro = SIMD.float64x2.splat(this.ro);
        let rom = SIMD.float64x2.splat(1-this.ro);
        let eps = SIMD.float64x2.splat(this.eps);

        // initialize lists for accumulators. Will only be done once on first iteration
        if(this.gsum.length === 0) {
          // adadelta needs gsum and xsum
          for(let i = 0; i < pglist.length; i++) {
            this.gsum.push(new Float64Array(pglist[i].params.length));
            this.xsum.push(new Float64Array(pglist[i].params.length));
          }
        }

        let pglen = (pglist.length|0);

        // perform an update for all sets of weights
        for(let i = 0; i < pglen; i++) {
          // param, gradient, other options in future (custom learning rate etc)
          let {p, g, l2_decay_mul, l1_decay_mul} = pglist[i];

          // learning rate for some parameters.
          let l2_decay = SIMD.float64x2.splat(this.l2_decay * (l2_decay_mul || 1.0));
          let l1_decay = SIMD.float64x2.splat(this.l1_decay * (l1_decay_mul || 1.0));
          let gsumi = this.gsum[i];
          let xsumi = this.xsum[i];
          let plen = (p.length|0); 

          /*
           * SIMD spaghetti code ahead - best served warm and w/ bolognaise sauce.
           */

          for(let j = 0; j < plen; j += 2) {

            let pj = SIMD.float64x2.load(p, j);
            let gj = SIMD.float64x2.load(g, j);
            let gsumij = SIMD.float64x2.load(gsumi, j);
            let xsumij = SIMD.float64x2.load(xsumi, j);

            // accumulate weight decay loss
            l2_decay_loss = SIMD.float64x2.add(l2_decay_loss, SIMD.float64x2.div(SIMD.float64x2.mul(l1_decay, SIMD.float64x2.mul(pj, pj)), SIMD.float64x2.splat(2)));
            l1_decay_loss = SIMD.float64x2.add(l1_decay_loss, SIMD.mul(l1_decay, SIMD.float64x2.abs(pj)));

            let l1grad = SIMD.float64x2.mul(l1_decay, SIMD.float64x2.greaterThan(pj, SIMD.float64x2.splat(0.0)));
            let l2grad = SIMD.float64x2.mul(l2_decay, pj)

            let gij = SIMD.float64x2.add(l2grad, SIMD.float64x2.add(l1grad, gj)) // raw batch gradient
  		    
            gsumij = SIMD.float64x2.add(SIMD.float64x2.mul(ro, gsumij), SIMD.float64x2.mul(rom, SIMD.float64x2.mul(gij, gij)));
            let dx = SIMD.float64x2.neg(SIMD.float64x2.mul(SIMD.float64x2.sqrt(SIMD.float64x2.div(SIMD.float64x2.add(xsumij, eps), SIMD.float64x2.add(gsumij, eps))), gij));
            xsumij = SIMD.float64x2.add(SIMD.float64x2.mul(ro, gsumij), SIMD.float64x2.mul(rom, SIMD.float64x2.mul(dx, dx)));
            // yes, xsum lags behind gsum by 1.

            SIMD.float64x2.store(gsumi, j, gsumij);
            SIMD.float64x2.store(xsumi, j, xsumij);

            SIMD.float64x2.store(p, j, SIMD.float64x2.add(pj, dx));
            
            // zero out gradient so that we can begin accumulating anew
            SIMD.float64x2.store(g, j, SIMD.float64x2.zero());

          }
        
        }

        l2DecayLoss = l2_decay_loss.x + l2_decay_loss.y + l2_decay_loss.z + l2_decay_loss.w;
        l1DecayLoss = l1_decay_loss.x + l1_decay_loss.y + l1_decay_loss.z + l1_decay_loss.w;
    
    }

    // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
    // in future, TODO: have to completely redo the way loss is done around the network as currently 
    // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
    // and it should all be computed correctly and automatically. 
    return {
      fwd_time: fwd_time, 
      bwd_time: bwd_time,
      l2_decay_loss: l2DecayLoss, 
      l1_decay_loss: l1DecayLoss,
      cost_loss: cost_loss, 
      softmax_loss: cost_loss, 
      loss: cost_loss + l1DecayLoss + l2DecayLoss
    }

	}

}
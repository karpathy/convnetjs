import * as Trainer from "./trainer.js";

export default class AdadeltaTrainer extends Trainer {
	
	constructor(options = {}){
		super(options);
    this.xsum = [];
    this.ro = options.ro || 0.95; // used in adadelta
    this.eps = options.eps || 1e-6; // used in adadelta
	}

	train(x, y){

		var start_1 = new Date().getTime();
    this.net.forward(x, true); // also set the flag that lets the net know we're just training
    var fwd_time = (new Date().getTime()) - start;

    var start_2 = new Date().getTime();
    var cost_loss = this.net.backward(y);
    var bwd_time = (new Date().getTime()) - start;

    var l2_decay_loss = 0.0;
    var l1_decay_loss = 0.0; 
      
    this.k++;
    if(this.k % this.batch_size === 0) {

      var pglist = this.net.getParamsAndGrads();

      // initialize lists for accumulators. Will only be done once on first iteration
      if(this.gsum.length === 0) {
        // adadelta needs gsum and xsum
        for(var i = 0; i < pglist.length; i++) {
          this.gsum.push(new Float64Array(pglist[i].params.length));
          this.xsum.push(new Float64Array(pglist[i].params.length));
        }
      }

      let pglen = (pglist.length|0);

      // perform an update for all sets of weights
      for(var i = 0; i < pglen; i++) {
        let {p, g, ...pg} = pglist[i]; // param, gradient, other options in future (custom learning rate etc)

        // learning rate for some parameters.
        let l2_decay = this.l2_decay * (pg.l2_decay_mul || 1.0);
        let l1_decay = this.l1_decay * (pg.l1_decay_mul || 1.0);

        let plen = (p.length|0);

        for(var j = 0; j < plen; j++) {

          l2_decay_loss += +(l2_decay*p[j]*p[j]/2); // accumulate weight decay loss
          l1_decay_loss += +(l1_decay*Math.abs(p[j]));
          let l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
          let l2grad = l2_decay * (p[j]);

          let gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

          let gsumi = this.gsum[i];
          let xsumi = this.xsum[i];
		    
          gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
          let dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
          xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
          p[j] += dx;
          
          g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
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
      l2_decay_loss: l2_decay_loss, 
      l1_decay_loss: l1_decay_loss,
      cost_loss: cost_loss, 
      softmax_loss: cost_loss, 
      loss: cost_loss + l1_decay_loss + l2_decay_loss
    };

	}

}
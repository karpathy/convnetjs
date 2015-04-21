import * as Trainer from "./trainer.js";

export default class SGDTrainer extends Trainer {
	
	constructor(opts = {}){
		super(opts);
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
      if(this.gsum.length === 0 && this.momentum > 0.0) {
        // only vanilla sgd doesnt need either lists
        // momentum needs gsum
        for(var i=0;i<pglist.length;i++) {
          this.gsum.push(new Float64Array(pglist[i].params.length));
        }
      }

      // perform an update for all sets of weights
      for(var i=0;i<pglist.length;i++) {
        var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
        var p = pg.params;
        var g = pg.grads;

        // learning rate for some parameters.
        var l2_decay_mul = pg.l2_decay_mul || 1.0;
        var l1_decay_mul = pg.l1_decay_mul || 1.0;
        var l2_decay = this.l2_decay * l2_decay_mul;
        var l1_decay = this.l1_decay * l1_decay_mul;

        var plen = p.length;
        for(var j=0;j<plen;j++) {
          l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
          l1_decay_loss += l1_decay*Math.abs(p[j]);
          var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
          var l2grad = l2_decay * (p[j]);

          var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

          var gsumi = this.gsum[i];
          var xsumi = this.xsum[i];
		    
	        if(this.momentum > 0.0) {
	          // momentum update
	          var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
	          gsumi[j] = dx; // back this up for next iteration of momentum
	          p[j] += dx; // apply corrected gradient
	        } else {
	          // vanilla sgd
	          p[j] +=  - this.learning_rate * gij;
	        }
          
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
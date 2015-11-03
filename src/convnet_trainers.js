var Buckets = require('buckets-js');

(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  /**
   * mode:
   *    normal    : classic training method
   *    hardcore  : keep training on the failed ones 10 times or until cost = 0
   *    
   */
  var Trainer = function(net, options) {

    this.net = net;

    var options = options || {};
    // enhanced params
    this.mode = typeof options.mode !== 'undefined' ? options.mode : "normal";
    this.costThreshold = typeof options.costThreshold !== 'undefined' ? options.costThreshold : 0;
    
    // basic training params
    this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
    this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
    this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
    this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
    this.method = typeof options.method !== 'undefined' ? options.method : 'sgd'; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

    this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;
    this.ro = typeof options.ro !== 'undefined' ? options.ro : 0.95; // used in adadelta
    this.eps = typeof options.eps !== 'undefined' ? options.eps : 1e-8; // used in adam or adadelta
    this.beta1 = typeof options.beta1 !== 'undefined' ? options.beta1 : 0.9; // used in adam
    this.beta2 = typeof options.beta2 !== 'undefined' ? options.beta2 : 0.999; // used in adam

    this.k = 0; // iteration counter
    this.gsum = []; // last iteration gradients (used for momentum calculations)
    this.xsum = []; // used in adam or adadelta
    
    // constant
    this.maxPQueueSize = 5;
    this.maxPQueueRetry = 20;
    
    // init
    this.costPQueue =  new Buckets.PriorityQueue(function(sampleA, sampleB){
      if(sampleA.priority < sampleB.priority){
        return -1;
      }else if(sampleA.priority > sampleB.priority){
        return 1;
      }else{
        return 0;
      }
    });
  }

  Trainer.prototype = {
    train: function(x, y) {
      this.net.forward(x, true);
      
      var trainStatus = this.trainCore(x,y);
      
      if(this.mode == "hardcore"){
        var cost = trainStatus.cost_loss;
        
        if(cost > this.costThreshold && this.costPQueue.size() <= this.maxPQueueSize){
          this.costPQueue.add({data:{x:x, y:y}, priority:cost}); //using inverse cost
        }
        
        var iter = 0;
        while(!this.costPQueue.isEmpty() && iter<this.maxPQueueRetry){
          var trainingPair = this.costPQueue.dequeue().data;
          trainStatus = this.trainCore(trainingPair.x, trainingPair.y);
          cost = trainStatus.cost_loss;
          if(cost > this.costThreshold){
            // push the sample back
            this.costPQueue.add({data:trainingPair, priority:cost});
          }
          iter++;
        }
        
      }
      
      return trainStatus; // return the highest cost status
    },
    
    
    trainCore: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        var pglist = this.net.getParamsAndGrads();

        // initialize lists for accumulators. Will only be done once on first iteration
        if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
          // only vanilla sgd doesnt need either lists
          // momentum needs gsum
          // adagrad needs gsum
          // adam and adadelta needs gsum and xsum
          for(var i=0;i<pglist.length;i++) {
            this.gsum.push(global.zeros(pglist[i].params.length));
            if(this.method === 'adam' || this.method === 'adadelta') {
              this.xsum.push(global.zeros(pglist[i].params.length));
            } else {
              this.xsum.push([]); // conserve memory
            }
          }
        }

        // perform an update for all sets of weights
        for(var i=0;i<pglist.length;i++) {
          var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
          var p = pg.params;
          var g = pg.grads;

          // learning rate for some parameters.
          var l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
          var l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
          
          
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
            if(this.method === 'adam') {
              // adam update
              gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
              xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
              var biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
              var biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
              var dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
              p[j] += dx;
            } else if(this.method === 'adagrad') {
              // adagrad update
              gsumi[j] = gsumi[j] + gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
              p[j] += dx;
            } else if(this.method === 'windowgrad') {
              // this is adagrad but with a moving window weighted average
              // so the gradient is not accumulated over the entire history of the run. 
              // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
              p[j] += dx;
            } else if(this.method === 'adadelta') {
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
              xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
              p[j] += dx;
            } else if(this.method === 'nesterov') {
            	var dx = gsumi[j];
            	gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                p[j] += dx;
            } else {
              // assume SGD
              if(this.momentum > 0.0) {
                // momentum update
                var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                gsumi[j] = dx; // back this up for next iteration of momentum
                p[j] += dx; // apply corrected gradient
              } else {
                // vanilla sgd
                p[j] +=  - this.learning_rate * gij;
              }
            }
            g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
          }
        }
      }

      // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
      // in future, TODO: have to completely redo the way loss is done around the network as currently 
      // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
      // and it should all be computed correctly and automatically. 
      return {fwd_time: fwd_time, bwd_time: bwd_time, 
              l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
              cost_loss: cost_loss, softmax_loss: cost_loss, 
              loss: cost_loss + l1_decay_loss + l2_decay_loss}
    }
  }
  
  global.Trainer = Trainer;
  global.SGDTrainer = Trainer; // backwards compatibility
})(convnetjs);


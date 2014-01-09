(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  var SGDTrainer = function(net, options) {

    this.net = net;

    var options = options || {};
    this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
    this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
    this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
    this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
    this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;

    if(typeof options.momentum !== 'undefined') this.momentum = options.momentum;
    this.k = 0; // iteration counter

    this.last_gs = []; // last iteration gradients (used for momentum calculations)
  }

  SGDTrainer.prototype = {
    train: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      var start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      var end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        // initialize lists for momentum keeping. Will only run first iteration
        var pglist = this.net.getParamsAndGrads();
        if(this.last_gs.length === 0 && this.momentum > 0.0) {
          for(var i=0;i<pglist.length;i++) {
            this.last_gs.push(global.zeros(pglist[i].params.length));
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
            if(this.momentum > 0.0) {
              // back up the last gradients and do weighted update
              var dir = -this.learning_rate * (l2grad + l1grad + g[j]) / this.batch_size;
              var dir_adj = this.momentum * this.last_gs[i][j] + (1.0 - this.momentum) * dir;
              p[j] += dir_adj;
              this.last_gs[i][j] = dir_adj;
            } else {
              // vanilla sgd
              p[j] -= this.learning_rate * (l2grad + l1grad + g[j]) / this.batch_size;
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
  
  global.SGDTrainer = SGDTrainer;
})(convnetjs);


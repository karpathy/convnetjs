import * as Trainers from "../trainers/index.js";
import Net from "./net.js";
import {EventEmitter} from "events";

/*
  A MagicNet takes data: a list of convnetjs.Vol(), and labels
  which for now are assumed to be class indeces 0..K. MagicNet then:
  - creates data folds for cross-validation
  - samples candidate networks
  - evaluates candidate networks on all data folds
  - produces predictions by model-averaging the best networks
*/

export default class MagicNet extends EventEmitter {

  constructor(data = [], labels = [], opt = {}){
    super()

    // required inputs
    this.data = data; // store these pointers to data
    this.labels = labels;

    // optional inputs
    this.train_ratio = opt.train_ratio || 0.7;
    this.num_folds = opt.num_folds || 10;
    this.num_candidates = opt.num_candidates || 50; // we evaluate several in parallel
    // how many epochs of data to train every network? for every fold?
    // higher values mean higher accuracy in final results, but more expensive
    this.num_epochs = opt.num_epochs || 50; 
    // number of best models to average during prediction. Usually higher = better
    this.ensemble_size = opt.ensemble_size || 10;

    // candidate parameters
    this.batch_size_min = opt.batch_size_min || 10;
    this.batch_size_max = opt.batch_size_max || 300;
    this.l2_decay_min = opt.l2_decay_min || -4;
    this.l2_decay_max = opt.l2_decay_max || 2;
    this.learning_rate_min = opt.learning_rate_min || -4;
    this.learning_rate_max = opt.learning_rate_max || 0;
    this.momentum_min = opt.momentum_min || 0.9;
    this.momentum_max = opt.momentum_max || 0.9;
    this.neurons_min = opt.neurons_min || 5;
    this.neurons_max = opt.neurons_max || 30;

    // computed
    this.folds = []; // data fold indices, gets filled by sampleFolds()
    this.candidates = []; // candidate networks that are being currently evaluated
    this.evaluated_candidates = []; // history of all candidates that were fully evaluated on all folds
    this.unique_labels = labels.filter((x, i) => {return (arr.indexOf(x) >= i)});
    this.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
    this.foldix = 0; // index of active fold

    // initializations
    if(this.data.length > 0) {
      this.sampleFolds();
      this.sampleCandidates();
    }

  }

  // sets this.folds to a sampling of this.num_folds folds
  sampleFolds() {
    var N = this.data.length;
    var num_train = Math.floor(this.train_ratio * N);
    this.folds = []; // flush folds, if any
    for(var i=0;i<this.num_folds;i++) {
      let p = [];
      var i = n,
          j = 0,
          temp;
      for(var q = 0; q < n; q++){ 
        j[q] = q;
      }
      while (i--) {
          j = Math.floor(Math.random() * (i+1));
          temp = j[i];
          j[i] = j[j];
          j[j] = temp;
      }
      this.folds.push({train_ix: p.slice(0, num_train), test_ix: p.slice(num_train, N)});
    }
  }

  // returns a random candidate network
  sampleCandidate() {
    var input_depth = this.data[0].w.length;
    var num_classes = this.unique_labels.length;

    // sample network topology and hyperparameters
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: input_depth});

    let nl; // prefer nets with 1,2 hidden layers
    {
      let lst = [0,1,2,3];
      let probs = [0.2, 0.3, 0.3, 0.2];
      let p = Math.random(); 
      let cumprob = 0.0;
      for(let k = 0, n = lst.length; k < n; k++) {
        cumprob += probs[k];
        if(p < cumprob) { 
          nl = lst[k]; 
        }
      }
    }

    for(var q=0;q<nl;q++) {
      var ni = randi(this.neurons_min, this.neurons_max);
      var act = ['tanh','maxout','relu'][randi(0,3)];
      if(randf(0,1)<0.5) {
        var dp = Math.random();
        layer_defs.push({type:'fc', num_neurons: ni, activation: act, drop_prob: dp});
      } else {
        layer_defs.push({type:'fc', num_neurons: ni, activation: act});
      }
    }
    layer_defs.push({type:'softmax', num_classes: num_classes});
    var net = new Net(layer_defs);

    // sample training hyperparameters
    var bs = randi(this.batch_size_min, this.batch_size_max); // batch size
    var l2 = Math.pow(10, randf(this.l2_decay_min, this.l2_decay_max)); // l2 weight decay
    var lr = Math.pow(10, randf(this.learning_rate_min, this.learning_rate_max)); // learning rate
    var mom = Math.random() * (this.momentum_max - this.momentum_min) + this.moment_min; // momentum. Lets just use 0.9, works okay usually ;p
    var tp = Math.random(); // trainer type
    let trainer_def;
    let trainer;
    if(tp<0.33) {
      trainer = new Trainers.AdadeltaTrainer(net, {batch_size:bs, l2_decay:l2});
      trainer_def = {name : 'Adadelta', batch_size:bs, l2_decay:l2};
    } else if(tp<0.66) {
      trainer = new Trainers.AdagradTrainer(net, {learning_rate: lr, batch_size:bs, l2_decay:l2});
      trainer_def = {name : 'Adagrad', learning_rate: lr, batch_size:bs, l2_decay:l2}
    } else {
      trainer = new Trainers.SGDTrainer(net, {learning_rate: lr, momentum: mom, batch_size:bs, l2_decay:l2});
      trainer_def = {name: 'SGD', learning_rate: lr, momentum: mom, batch_size:bs, l2_decay:l2};
    }

    return {
      acc : [],
      accv : 0,
      layer_defs : layer_defs,
      trainer_def : trainer_def,
      net : net,
      trainer : trainer
    };
  }

  // sets this.candidates with this.num_candidates candidate nets
  sampleCandidates() {
    this.candidates = []; // flush, if any
    for(var i=0;i<this.num_candidates;i++) {
      var cand = this.sampleCandidate();
      this.candidates.push(cand);
    }
  }

  step() {
    
    // run an example through current candidate
    this.iter++;

    // step all candidates on a random data point
    var fold = this.folds[this.foldix]; // active fold
    var dataix = fold.train_ix[randi(0, fold.train_ix.length)];
    for(var k=0;k<this.candidates.length;k++) {
      var x = this.data[dataix];
      var l = this.labels[dataix];
      this.candidates[k].trainer.train(x, l);
    }

    // process consequences: sample new folds, or candidates
    var lastiter = this.num_epochs * fold.train_ix.length;
    if(this.iter >= lastiter) {
      // finished evaluation of this fold. Get final validation
      // accuracies, record them, and go on to next fold.
      var val_acc = this.evalValErrors();
      for(var k=0;k<this.candidates.length;k++) {
        var c = this.candidates[k];
        c.acc.push(val_acc[k]);
        c.accv += val_acc[k];
      }
      this.iter = 0; // reset step number
      this.foldix++; // increment fold

      this.emit('finishfold');

      if(this.foldix >= this.folds.length) {
        // we finished all folds as well! Record these candidates
        // and sample new ones to evaluate.
        for(var k=0;k<this.candidates.length;k++) {
          this.evaluated_candidates.push(this.candidates[k]);
        }
        // sort evaluated candidates according to accuracy achieved
        this.evaluated_candidates.sort(function(a, b) { 
          return (a.accv / a.acc.length) 
               > (b.accv / b.acc.length) 
               ? -1 : 1;
        });
        // and clip only to the top few ones (lets place limit at 3*ensemble_size)
        // otherwise there are concerns with keeping these all in memory 
        // if MagicNet is being evaluated for a very long time
        if(this.evaluated_candidates.length > 3 * this.ensemble_size) {
          this.evaluated_candidates = this.evaluated_candidates.slice(0, 3 * this.ensemble_size);
        }
        
        this.emit('finishbatch');

        this.sampleCandidates(); // begin with new candidates
        this.foldix = 0; // reset this
      } else {
        // we will go on to another fold. reset all candidates nets
        for(var k=0;k<this.candidates.length;k++) {
          let c = this.candidates[k];
          let net = new Net(c.layer_defs);
          let trainer = new Trainers[c.trainer_def.name + "Trainer"](net, c.trainer_def);
          c.net = net;
          c.trainer = trainer;
        }
      }
    }
  }

  evalValErrors() {
    // evaluate candidates on validation data and return performance of current networks
    // as simple list
    var vals = [];
    var fold = this.folds[this.foldix]; // active fold
    for(var k = 0; k < this.candidates.length; k++) {
      var net = this.candidates[k].net;
      var v = 0.0;
      for(var q = 0; q < fold.test_ix.length; q++) {
        var x = this.data[fold.test_ix[q]];
        var l = this.labels[fold.test_ix[q]];
        net.forward(x);
        var yhat = net.getPrediction();
        v += (yhat === l ? 1.0 : 0.0); // 0 1 loss
      }
      v /= fold.test_ix.length; // normalize
      vals.push(v);
    }
    return vals;
  }

  // returns prediction scores for given test data point, as Vol
  // uses an averaged prediction from the best ensemble_size models
  // x is a Vol.
  predictSoft(data) {
    // forward prop the best networks
    // and accumulate probabilities at last layer into a an output Vol

    var eval_candidates = [];
    var nv = 0;
    if(this.evaluated_candidates.length === 0) {
      // not sure what to do here, first batch of nets hasnt evaluated yet
      // lets just predict with current candidates.
      nv = this.candidates.length;
      eval_candidates = this.candidates;
    } else {
      // forward prop the best networks from evaluated_candidates
      nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
      eval_candidates = this.evaluated_candidates
    }

    // forward nets of all candidates and average the predictions
    var xout, n;
    for(var j=0;j<nv;j++) {
      var net = eval_candidates[j].net;
      var x = net.forward(data);
      if(j===0) { 
        xout = x; 
        n = x.w.length; 
      } else {
        // add it on
        for(var d=0;d<n;d++) {
          xout.w[d] += x.w[d];
        }
      }
    }
    // produce average
    for(var d=0;d<n;d++) {
      xout.w[d] /= nv;
    }
    return xout;
  }

  predict(data) {
    var xout = this.predict_soft(data);
    let w = new Float64Array(TypedObject.storage(xout.w).buffer);
    let maxv = w[0];
    let maxi = 0;
    let n = w.length;
    for(let i = 1; i < n; i++) {
      if(w[i] > maxv) { 
        maxi = i; 
      } 
    }
    return maxi;
  }

  toJSON() {
    // dump the top ensemble_size networks as a list
    var nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
    var json = {};
    json.nets = [];
    for(var i=0;i<nv;i++) {
      json.nets.push(this.evaluated_candidates[i].net.toJSON());
    }
    return json;
  }

  static fromJSON(json) {
    this.ensemble_size = json.nets.length;
    this.evaluated_candidates = [];
    for(var i=0;i<this.ensemble_size;i++) {
      var net = new Net();
      net.fromJSON(json.nets[i]);
      var dummy_candidate = {};
      dummy_candidate.net = net;
      this.evaluated_candidates.push(dummy_candidate);
    }
  }

}
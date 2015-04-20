export class Trainer {

  constructor(net, options = {}){
    this.net = net;

    this.learning_rate = options.learning_rate || 0.01;
    this.l1_decay = options.l1_decay || 0.0;
    this.l2_decay = options.l2_decay || 0.0;
    this.batch_size = options.batch_size || 1;
    this.method = options.method || 'sgd'; // sgd/adagrad/adadelta/windowgrad/netsterov

    this.momentum = options.momentum || 0.9;
    this.ro = options.ro || 0.95; // used in adadelta
    this.eps = options.eps || 1e-6; // used in adadelta

    this.k = 0; // iteration counter
    this.gsum = []; // last iteration gradients (used for momentum calculations)
    this.xsum = []; // used in adadelta
  }
  
}
var assert = require('assert');
var convnetjs = require('../index.js');

// tanh are their own layers. Softmax gets its own fully connected layer.
// this should all get desugared just fine.

{
	let net = new convnetjs.Net();

    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    net.makeLayers(layer_defs);

    let trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});

	assert.equal(net.layers.length, 7, "It wasn't possible to initialize a Net.");
}


{

	let net = new convnetjs.Net();

    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    net.makeLayers(layer_defs);

    let trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});

	var x = new (new convnetjs.VolType(1,1,2))({w:[[[0.2, -0.3]]]});
    var probability_volume = net.forward(x);

    assert.equal(probability_volume.w.length, 3); // 3 classes output
    var w = probability_volume.w;
    for(var i=0;i<3;i++) {
      expect(w[i]).toBeGreaterThan(0);
      expect(w[i]).toBeLessThan(1.0);
    }

    expect(w[0]+w[1]+w[2]).toBeCloseTo(1.0);

}

// lets test 100 random point and label settings
// note that this should work since l2 and l1 regularization are off
// an issue is that if step size is too high, this could technically fail...

{
	let net = new convnetjs.Net();

    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    net.makeLayers(layer_defs);

    let trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});

    for(let k=0;k<100;k++) {
	    let x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
	    let pv = net.forward(x);
	    let gti = Math.floor(Math.random() * 3);
	    trainer.train(x, gti);
	    let pv2 = net.forward(x);
	    expect(pv2.w[gti]).toBeGreaterThan(pv.w[gti]);
	  }

	assert(pv2.w[gti] > pv.w[gti], "Didn't increase probabilities for ground truth class when trained.");
}

// here we only test the gradient at data, but if this is
// right then that's comforting, because it is a function 
// of all gradients above, for all layers.

{

	let net = new convnetjs.Net();

    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    net.makeLayers(layer_defs);

    let trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});

	let V = new (new convnetjs.VolType(1,1,2))({w:[[[Math.random() * 2 - 1, Math.random() * 2 - 1]]]});
    let gti = Math.floor(Math.random() * 3); // ground truth index
    trainer.train(V, gti); // computes gradients at all layers, and at x

    let delta = 0.000001;

    for(let x = 0; x < V.w.length; x++) {

      for(let y = 0; y < V.w[0].length; y++){

        for(let d = 0; y < V.w[0][0].length; d++){

          let grad_analytic = V.dw[x][y][d];

          let xold = V.w[x][y][d];
          V.w[x][y][d] += delta;
          let c0 = net.getCostLoss(V, gti);
          V.w[x][y][d] -= 2*delta;
          let c1 = net.getCostLoss(V, gti);
          V.w[x][y][d] = xold; // reset

          let grad_numeric = (c0 - c1)/(2 * delta);
          let rel_error = Math.abs(grad_analytic - grad_numeric)/Math.abs(grad_analytic + grad_numeric);
          console.log(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);
          
          assert(rel_error < 1e-2, "Couldn't compute correct gradient at data.");

        }

      }    

    }

}
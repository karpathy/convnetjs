describe("Simple Fully-Connected Neural Net Classifier", function() {
  var net;
  var trainer;

  beforeEach(function() {
    net = new convnetjs.Net();

    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    net.makeLayers(layer_defs);

    trainer = new convnetjs.SGDTrainer(net, 
          {learning_rate:0.0001, momentum:0.0, batch_size:1, l2_decay:0.0});
  });

  it("should be possible to initialize", function() {
    
    // tanh are their own layers. Softmax gets its own fully connected layer.
    // this should all get desugared just fine.
    expect(net.layers.length).toEqual(7); 
    
  });

  it("should forward prop volumes to probabilities", function() {

    var x = new convnetjs.Vol([0.2, -0.3]);
    var probability_volume = net.forward(x);

    expect(probability_volume.w.length).toEqual(3); // 3 classes output
    var w = probability_volume.w;
    for(var i=0;i<3;i++) {
      expect(w[i]).toBeGreaterThan(0);
      expect(w[i]).toBeLessThan(1.0);
    }
    expect(w[0]+w[1]+w[2]).toBeCloseTo(1.0);

  });

  it("should increase probabilities for ground truth class when trained", function() {

    // lets test 100 random point and label settings
    // note that this should work since l2 and l1 regularization are off
    // an issue is that if step size is too high, this could technically fail...
    for(var k=0;k<100;k++) {
      var x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
      var pv = net.forward(x);
      var gti = Math.floor(Math.random() * 3);
      trainer.train(x, gti);
      var pv2 = net.forward(x);
      expect(pv2.w[gti]).toBeGreaterThan(pv.w[gti]);
    }

  });

  it("should compute correct gradient at data", function() {

    // here we only test the gradient at data, but if this is
    // right then that's comforting, because it is a function 
    // of all gradients above, for all layers.

    var x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
    var gti = Math.floor(Math.random() * 3); // ground truth index
    trainer.train(x, gti); // computes gradients at all layers, and at x

    var delta = 0.000001;

    for(var i=0;i<x.w.length;i++) {

      var grad_analytic = x.dw[i];

      var xold = x.w[i];
      x.w[i] += delta;
      var c0 = net.getCostLoss(x, gti);
      x.w[i] -= 2*delta;
      var c1 = net.getCostLoss(x, gti);
      x.w[i] = xold; // reset

      var grad_numeric = (c0 - c1)/(2 * delta);
      var rel_error = Math.abs(grad_analytic - grad_numeric)/Math.abs(grad_analytic + grad_numeric);
      console.log(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);
      expect(rel_error).toBeLessThan(1e-2);

    }
  });
});

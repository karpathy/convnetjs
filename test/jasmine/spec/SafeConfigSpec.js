describe("Safe demo configuration parser", function() {
  it("should create a network and trainer from the demo configuration dialect", function() {
    var source = "\n\
      layer_defs = [];\n\
      layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});\n\
      layer_defs.push({type:'fc', num_neurons:6, activation:'tanh'});\n\
      layer_defs.push({type:'softmax', num_classes:2});\n\
      net = new convnetjs.Net();\n\
      net.makeLayers(layer_defs);\n\
      trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.1, batch_size:10, l2_decay:0.001});\n\
    ";

    var config = convnetjs.demoConfig.parseNetwork(source);

    expect(config.layer_defs.length).toEqual(3);
    expect(config.net.layers.length).toEqual(5);
    expect(config.trainer.learning_rate).toEqual(0.01);
    expect(config.trainer.batch_size).toEqual(10);
  });

  it("should resolve constants and arithmetic expressions without executing JavaScript", function() {
    var source = "\n\
      var depth = 2 * 3 + 1;\n\
      layer_defs = [];\n\
      layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:depth});\n\
      layer_defs.push({type:'regression', num_neurons:28*28});\n\
      net = new convnetjs.Net();\n\
      net.makeLayers(layer_defs);\n\
      trainer = new convnetjs.SGDTrainer(net, {learning_rate:1, method:'adadelta', batch_size:50});\n\
    ";

    var config = convnetjs.demoConfig.parseNetwork(source);

    expect(config.layer_defs[0].out_depth).toEqual(7);
    expect(config.layer_defs[1].num_neurons).toEqual(784);
    expect(config.trainer.method).toEqual("adadelta");
  });

  it("should parse trainer comparison configs", function() {
    var source = "\n\
      var layer_defs = [];\n\
      layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});\n\
      layer_defs.push({type:'softmax', num_classes:10});\n\
      var LR = 0.01;\n\
      var BS = 8;\n\
      var L2 = 0.001;\n\
      trainer_defs = [];\n\
      trainer_defs.push({learning_rate:LR, method:'sgd', momentum:0.0, batch_size:BS, l2_decay:L2});\n\
      trainer_defs.push({learning_rate:LR, method:'adam', eps:1e-8, beta1:0.9, beta2:0.99, batch_size:BS, l2_decay:L2});\n\
      legend = ['sgd', 'adam'];\n\
    ";

    var config = convnetjs.demoConfig.parseTrainerComparison(source);

    expect(config.layer_defs.length).toEqual(2);
    expect(config.trainer_defs.length).toEqual(2);
    expect(config.trainer_defs[1].method).toEqual("adam");
    expect(config.legend[0]).toEqual("sgd");
  });

  it("should reject unsupported executable statements", function() {
    expect(function() {
      convnetjs.demoConfig.parseNetwork("window.alert('xss');");
    }).toThrow();
  });
});

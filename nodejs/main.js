var fs = require('fs');
var util = require("util");

var PNG = require('pngjs2').PNG;
var convnetjs = require("convnetjs");

/**
 * Node js Cat Learner
 * @augments
 *  -i :    the number of iterations to run; default 10
 *  -b :    batches per iteration(i.e. number of times to train with `batch_size` data points).
 *  -o :    output file name.
 *  -p :    output PNG filename.
 *  -n :    input neural network definition file. This is a json file containing an array of 
 *          network layers. Default a 9 layer network that takes a 2 dimensional 
 *          input vector(i.e. x,y coordinates) and outputs a 3 dimensional vector(i.e. RGV)
 */
var args = process.argv.slice(2);

function check_argument(flag) {
    let flag_idx = args.indexOf(flag);
    return (flag_idx === -1) ? false : args[flag_idx + 1];
}

/**
 * 
 */
var iteration_count = check_argument('-i') || 10;
var batches_per_iteration = check_argument('-b') || 100;

var mod_skip_draw = 100;
var smooth_loss = -1;

var output_file = check_argument('-o') || false;
if (output_file) {
    fs.writeFileSync(output_file, 'iteration,error\n');
}

var output_png = check_argument('-p') || false;
var network_definition = check_argument('-n') || false;

/**
 * Load the image of a cat and create an RBGalpha Array
 */
image_data = [];

var data = fs.readFileSync('cat.png');
var cat = PNG.sync.read(data);

for (var y = 0; y < cat.height; y++) {
    for (var x = 0; x < cat.width; x++) {
        var idx = (cat.width * y + x) << 2;

        image_data.push(cat.data[idx],
            cat.data[idx + 1],
            cat.data[idx + 2],
            cat.data[idx + 3]);
    }
}

// forward prop the data
var W = cat.width;
var H = cat.height;


var p = Uint8ClampedArray.from(image_data);
/**
 * Build the neural network
 */
layer_defs = [];

if (network_definition) {
    definition = JSON.parse(fs.readFileSync(network_definition, 'utf8'));

    for(var layer in definition){
        layer_defs.push(definition[layer]);
    }
       
} else {
    layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 }); // 2 inputs: x, y 
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'fc', num_neurons: 20, activation: 'relu' });
    layer_defs.push({ type: 'regression', num_neurons: 3 }); // 3 outputs: r,g,b 
}
var net = new convnetjs.Net();

net.makeLayers(layer_defs);

var trainer = new convnetjs.SGDTrainer(net, { learning_rate: 0.01, momentum: 0.9, batch_size: 5, l2_decay: 0.0 });

var counter = 0;
/**
 * @see https://github.com/karpathy/convnetjs/blob/master/demo/js/image_regression.js
 * 
 */
function update(iteration) {

    var v = new convnetjs.Vol(1, 1, 2);
    var loss = 0;
    var lossi = 0;

    var N = batches_per_iteration;

    for (var iters = 0; iters < trainer.batch_size; iters++) {
        for (var i = 0; i < N; i++) {
            // sample a coordinate
            var x = convnetjs.randi(0, W);
            var y = convnetjs.randi(0, H);
            var ix = ((W * y) + x) * 4;
            var r = [p[ix] / 255.0, p[ix + 1] / 255.0, p[ix + 2] / 255.0]; // r g b
            v.w[0] = (x - W / 2) / W;
            v.w[1] = (y - H / 2) / H;
            var stats = trainer.train(v, r);
            loss += stats.loss;
            lossi += 1;
        }
    }
    loss /= lossi;

    if (counter === 0) smooth_loss = loss;
    else smooth_loss = 0.99 * smooth_loss + 0.01 * loss;


    if (output_file) {
        fs.appendFileSync(output_file, util.format("%d,%s\n", iteration, smooth_loss));
    } else {
        console.log(smooth_loss);
    }
}


/**
 * `Main`
 * 
 */

if (require.main === module) {

    for (var i = 0; i < iteration_count; i++) {
        update(i);
    }

    if (output_png) {

        var g = new PNG({ width: 200, height: 200 });

        var v = new convnetjs.Vol(1, 1, 2);

        for (var x = 0; x < W; x++) {
            v.w[0] = (x - W / 2) / W;
            for (var y = 0; y < H; y++) {
                v.w[1] = (y - H / 2) / H;

                var ix = ((W * y) + x) * 4;
                var r = net.forward(v);

                g.data[ix + 0] = Math.floor(255 * r.w[0]);
                g.data[ix + 1] = Math.floor(255 * r.w[1]);
                g.data[ix + 2] = Math.floor(255 * r.w[2]);
                g.data[ix + 3] = 255;
            }
        }
        g.pack().pipe(fs.createWriteStream(output_png));
    }
}
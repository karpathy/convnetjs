import * as convnetjs from "../src/index";

import * as Chai from "chai";
const expect = Chai.expect;

describe("Simple Fully-Connected Neural Net Classifier", function () {
    let net: convnetjs.Net;
    let trainer: convnetjs.Trainer;

    beforeEach(function () {
        net = new convnetjs.Net();
        const layer_defs = [];
        layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 });
        layer_defs.push({ type: 'fc', num_neurons: 5, activation: 'tanh' });
        layer_defs.push({ type: 'fc', num_neurons: 5, activation: 'tanh' });
        layer_defs.push({ type: 'softmax', num_classes: 3 });
        net.makeLayers(layer_defs);
        trainer = new convnetjs.SGDTrainer(net,
            { learning_rate: 0.0001, momentum: 0.0, batch_size: 1, l2_decay: 0.0 });
    });

    it("should be possible to initialize", function () {

        // tanh are their own layers. Softmax gets its own fully connected layer.
        // this should all get desugared just fine.
        expect(net.layers.length).to.equal(7);

    });

    it("should forward prop volumes to probabilities", function () {

        const x = new convnetjs.Vol([0.2, -0.3]);
        const probability_volume = net.forward(x);

        expect(probability_volume.w.length).to.equal(3); // 3 classes output
        const w = probability_volume.w;
        for (let i = 0; i < 3; i++) {
            expect(w[i]).to.be.greaterThan(0);
            expect(w[i]).to.be.lessThan(1.0);
        }
        expect(w[0] + w[1] + w[2]).to.be.closeTo(1.0, 0.01);

    });

    it("should increase probabilities for ground truth class when trained", function () {

        // lets test 100 random point and label settings
        // note that this should work since l2 and l1 regularization are off
        // an issue is that if step size is too high, this could technically fail...
        for (let k = 0; k < 100; k++) {
            const x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
            const pv = net.forward(x);
            const gti = Math.floor(Math.random() * 3);
            trainer.train(x, gti);
            const pv2 = net.forward(x);
            expect(pv2.w[gti]).to.be.greaterThan(pv.w[gti]);
        }

    });

    it("should compute correct gradient at data", function () {

        // here we only test the gradient at data, but if this is
        // right then that's comforting, because it is a function
        // of all gradients above, for all layers.

        const x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
        const gti = Math.floor(Math.random() * 3); // ground truth index
        trainer.train(x, gti); // computes gradients at all layers, and at x

        const delta = 0.000001;

        for (let i = 0; i < x.w.length; i++) {

            const grad_analytic = x.dw[i];

            const xold = x.w[i];
            x.w[i] += delta;
            const c0 = net.getCostLoss(x, gti);
            x.w[i] -= 2 * delta;
            const c1 = net.getCostLoss(x, gti);
            x.w[i] = xold; // reset

            const grad_numeric = (c0 - c1) / (2 * delta);
            const rel_error = Math.abs(grad_analytic - grad_numeric) / Math.abs(grad_analytic + grad_numeric);
            console.log(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);
            expect(rel_error).to.be.lessThan(1e-2);

        }
    });
});

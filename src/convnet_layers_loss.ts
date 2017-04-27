import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

export interface LossLayerOptions extends LayerOptions {
    /** <required> */
    num_classes: number;
}

/** This is a classifier, with N discrete classes from 0 to N-1
 * it gets a stream of N incoming numbers and computes the softmax
 * function (exponentiate and normalize to sum to 1 as probabilities should)
 */
export class SoftmaxLayer extends LayerBase implements ILayer {
    in_act: Vol;
    num_inputs: number;
    out_act: Vol;
    es: number[] | Float64Array;

    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const lopt = <LossLayerOptions>opt;
        super(lopt);

        // computed
        this.num_inputs = <number>lopt.in_sx * <number>lopt.in_sy * <number>lopt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'softmax';
    }

    forward(V: Vol, ) {
        this.in_act = V;

        const A = new Vol(1, 1, this.out_depth, 0.0);

        // compute max activation
        const as = V.w;
        let amax = V.w[0];
        for (let i = 1; i < this.out_depth; i++) {
            if (as[i] > amax) { amax = as[i]; }
        }

        // compute exponentials (carefully to not blow up)
        const es = util.zeros(this.out_depth);
        let esum = 0.0;
        for (let i = 0; i < this.out_depth; i++) {
            const e = Math.exp(as[i] - amax);
            esum += e;
            es[i] = e;
        }

        // normalize and output to sum to one
        for (let i = 0; i < this.out_depth; i++) {
            es[i] /= esum;
            A.w[i] = es[i];
        }

        this.es = es; // save these for backprop
        this.out_act = A;
        return this.out_act;
    }
    backward(y: number) {

        // compute and accumulate gradient wrt weights and bias of this layer
        const x = this.in_act;
        x.dw = util.zeros(x.w.length); // zero out the gradient of input Vol

        for (let i = 0; i < this.out_depth; i++) {
            const indicator = i === y ? 1.0 : 0.0;
            const mul = -(indicator - this.es[i]);
            x.dw[i] = mul;
        }

        // loss is the class negative log likelihood
        return -Math.log(this.es[y]);
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON() {
        const json: LayerJSON = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.num_inputs = json.num_inputs as number;
    }
}

/**
 * implements an L2 regression cost layer,
 * so penalizes \sum_i(||x_i - y_i||^2), where x is its input
 * and y is the user-provided array of "correct" values.
 */
export class RegressionLayer extends LayerBase implements ILayer {
    num_inputs: number;
    in_act: Vol;
    out_act: Vol;


    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const lopt = <LossLayerOptions>opt;
        super(lopt);

        // computed
        this.num_inputs = <number>lopt.in_sx * <number>lopt.in_sy * <number>lopt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'regression';
    }
    forward(V: Vol, ) {
        this.in_act = V;
        this.out_act = V;
        return V; // identity function
    }
    // y is a list here of size num_inputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    backward(y: number | number[] | Float64Array | { [key: string]: number }) {

        // compute and accumulate gradient wrt weights and bias of this layer
        const x = this.in_act;
        x.dw = util.zeros(x.w.length); // zero out the gradient of input Vol
        let loss = 0.0;
        if (y instanceof Array || y instanceof Float64Array) {
            for (let i = 0; i < this.out_depth; i++) {
                const dy = x.w[i] - y[i];
                x.dw[i] = dy;
                loss += 0.5 * dy * dy;
            }
        } else if (typeof y === 'number') {
            // lets hope that only one number is being regressed
            const dy = x.w[0] - y;
            x.dw[0] = dy;
            loss += 0.5 * dy * dy;
        } else {
            // assume it is a struct with entries .dim and .val
            // and we pass gradient only along dimension dim to be equal to val
            const i = y.dim;
            const yi = y.val;
            const dy = x.w[i] - yi;
            x.dw[i] = dy;
            loss += 0.5 * dy * dy;
        }
        return loss;
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON() {
        const json: LayerJSON = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.num_inputs = json.num_inputs as number;
    }
}

export class SVMLayer extends LayerBase implements ILayer {
    num_inputs: number;
    in_act: Vol;
    out_act: Vol;


    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const lopt = <LossLayerOptions>opt;
        super(lopt);

        // computed
        this.num_inputs = <number>lopt.in_sx * <number>lopt.in_sy * <number>lopt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'svm';
    }

    forward(V: Vol, ) {
        this.in_act = V;
        this.out_act = V; // nothing to do, output raw scores
        return V;
    }
    backward(y: number) {

        // compute and accumulate gradient wrt weights and bias of this layer
        const x = this.in_act;
        x.dw = util.zeros(x.w.length); // zero out the gradient of input Vol

        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        const yscore = x.w[y]; // score of ground truth
        const margin = 1.0;
        let loss = 0.0;
        for (let i = 0; i < this.out_depth; i++) {
            if (y === i) { continue; }
            const ydiff = -yscore + x.w[i] + margin;
            if (ydiff > 0) {
                // violating dimension, apply loss
                x.dw[i] += 1;
                x.dw[y] -= 1;
                loss += ydiff;
            }
        }

        return loss;
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON() {
        const json: LayerJSON = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.num_inputs = json.num_inputs as number;
    }
}


import { Vol, VolJSON } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

// This file contains all layers that do dot products with input,
// but usually in a different connectivity pattern and weight sharing
// schemes:
// - FullyConn is fully connected dot products
// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar

export interface DotproductsLayerOptions extends LayerOptions {
    filters?: number;
    // optional
    /** <optional> add dropout layer with drop probability */
    drop_prob?: number;
    /** <optional> set activation function. */
    activation?: string;
    bias_pref?: number;
    l1_decay_mul?: number;
    l2_decay_mul?: number;
}

export interface ConvLayerOptions extends DotproductsLayerOptions {
    /** <required> */
    sx: number;
    /** <optional> */
    sy?: number;
    /** <optional> */
    stride?: number;
    /** <optional> */
    pad?: number;
    /** <optional> */
    l1_decay_mul?: number;
    /** <optional> */
    l2_decay_mul?: number;
}

export interface FullyConnLayerOptions extends DotproductsLayerOptions {
    num_neurons: number;
}

export class DotproductsLayer extends LayerBase {
    l1_decay_mul: number;
    l2_decay_mul: number;
    filters: Vol[];
    biases: Vol;
    out_depth: number;
    out_act: Vol;
    in_act: Vol;
    constructor(opt?: DotproductsLayerOptions) {
        if (!opt) { return; }
        super();

        // optional
        this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
        this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;
    }
}

/**
 * ConvLayer does convolutions (so weight sharing spatially)
*/
export class ConvLayer extends DotproductsLayer {
    sx: number;
    sy: number;
    stride: number;
    pad: number;
    in_depth: number;
    in_sx: number;
    in_sy: number;


    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const copt = <ConvLayerOptions>opt;
        super(copt);

        // required
        this.out_depth = copt.filters;
        this.sx = copt.sx; // filter size. Should be odd if possible, it's cleaner.
        this.in_depth = copt.in_depth as number;
        this.in_sx = copt.in_sx as number;
        this.in_sy = copt.in_sy as number;

        // optional
        this.sy = typeof copt.sy !== 'undefined' ? copt.sy : this.sx;
        this.stride = typeof copt.stride !== 'undefined' ? copt.stride : 1; // stride at which we apply filters to input volume
        this.pad = typeof copt.pad !== 'undefined' ? copt.pad : 0; // amount of 0 padding to add around borders of input volume

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = 'conv';

        // initializations
        this.filters = [];
        for (let i = 0; i < this.out_depth; i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
        const bias = <number>(typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0);
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    forward(V: Vol, ) {
        // optimized code by @mdda that achieves 2x speedup over previous version

        this.in_act = V;
        const A = new Vol(this.out_sx | 0, this.out_sy | 0, this.out_depth | 0, 0.0);

        const V_sx = V.sx | 0;
        const V_sy = V.sy | 0;
        const xy_stride = this.stride | 0;

        for (let d = 0; d < this.out_depth; d++) {
            const f = this.filters[d];
            let x = -this.pad | 0;
            let y = -this.pad | 0;
            for (let ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad | 0;
                for (let ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    let a = 0.0;
                    for (let fy = 0; fy < f.sy; fy++) {
                        const oy = y + fy; // coordinates in the original input array coordinates
                        for (let fx = 0; fx < f.sx; fx++) {
                            const ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (let fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((V_sx * oy) + ox) * V.depth + fd];
                                }
                            }
                        }
                    }
                    a += this.biases.w[d];
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return this.out_act;
    }
    backward() {

        const V = this.in_act;
        V.dw = util.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

        const V_sx = V.sx | 0;
        const V_sy = V.sy | 0;
        const xy_stride = this.stride | 0;

        for (let d = 0; d < this.out_depth; d++) {
            const f = this.filters[d];
            let x = -this.pad | 0;
            let y = -this.pad | 0;
            for (let ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                x = -this.pad | 0;
                for (let ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride

                    // convolve centered at this particular location
                    const chain_grad = this.out_act.get_grad(ax, ay, d); // gradient from above, from chain rule
                    for (let fy = 0; fy < f.sy; fy++) {
                        const oy = y + fy; // coordinates in the original input array coordinates
                        for (let fx = 0; fx < f.sx; fx++) {
                            const ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (let fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    const ix1 = ((V_sx * oy) + ox) * V.depth + fd;
                                    const ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                    f.dw[ix2] += V.w[ix1] * chain_grad;
                                    V.dw[ix1] += f.w[ix2] * chain_grad;
                                }
                            }
                        }
                    }
                    this.biases.dw[d] += chain_grad;
                }
            }
        }
        return 0;
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        const response = [] as ParamsAndGrads[];
        for (let i = 0; i < this.out_depth; i++) {
            response.push({ params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul });
        }
        response.push({ params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0 });
        return response;
    }
    toJSON() {
        const json: LayerJSON = {};
        json.sx = this.sx; // filter size in x, y dims
        json.sy = this.sy;
        json.stride = this.stride;
        json.in_depth = this.in_depth;
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.l1_decay_mul = this.l1_decay_mul;
        json.l2_decay_mul = this.l2_decay_mul;
        json.pad = this.pad;
        json.filters = [];
        for (let i = 0; i < this.filters.length; i++) {
            json.filters.push(this.filters[i].toJSON());
        }
        json.biases = this.biases.toJSON();
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.sx = json.sx as number; // filter size in x, y dims
        this.sy = json.sy as number;
        this.stride = json.stride as number;
        this.in_depth = json.in_depth as number; // depth of input volume
        this.filters = [];
        this.l1_decay_mul = (typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0) as number;
        this.l2_decay_mul = (typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0) as number;
        this.pad = (typeof json.pad !== 'undefined' ? json.pad : 0) as number;
        for (let i = 0; i < json.filters.length; i++) {
            const v = new Vol(0, 0, 0, 0);
            v.fromJSON(json.filters[i]);
            this.filters.push(v);
        }
        this.biases = new Vol(0, 0, 0, 0);
        this.biases.fromJSON(json.biases as VolJSON);
    }
}

/**
 * FullyConn is fully connected dot products
 */
export class FullyConnLayer extends DotproductsLayer implements ILayer {
    num_inputs: number;
    bias_pref: number;


    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const fcopt = <FullyConnLayerOptions>opt;
        super(fcopt);

        // required
        // ok fine we will allow 'filters' as the word as well
        this.out_depth = typeof fcopt.num_neurons !== 'undefined' ? fcopt.num_neurons : fcopt.filters;

        // computed
        this.num_inputs = <number>fcopt.in_sx * <number>fcopt.in_sy * <number>fcopt.in_depth;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'fc';

        // initializations
        this.filters = [];
        for (let i = 0; i < this.out_depth; i++) { this.filters.push(new Vol(1, 1, this.num_inputs)); }
        const bias = <number>(typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0);
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    forward(V: Vol, ) {
        this.in_act = V;
        const A = new Vol(1, 1, this.out_depth, 0.0);
        const Vw = V.w;
        for (let i = 0; i < this.out_depth; i++) {
            let a = 0.0;
            const wi = this.filters[i].w;
            for (let d = 0; d < this.num_inputs; d++) {
                a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
            }
            a += this.biases.w[i];
            A.w[i] = a;
        }
        this.out_act = A;
        return this.out_act;
    }
    backward() {
        const V = this.in_act;
        V.dw = util.zeros(V.w.length); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (let i = 0; i < this.out_depth; i++) {
            const tfi = this.filters[i];
            const chain_grad = this.out_act.dw[i];
            for (let d = 0; d < this.num_inputs; d++) {
                V.dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
                tfi.dw[d] += V.w[d] * chain_grad; // grad wrt params
            }
            this.biases.dw[i] += chain_grad;
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        const response = [];
        for (let i = 0; i < this.out_depth; i++) {
            response.push({ params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul });
        }
        response.push({ params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0 });
        return response;
    }
    toJSON() {
        const json: LayerJSON = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        json.l1_decay_mul = this.l1_decay_mul;
        json.l2_decay_mul = this.l2_decay_mul;
        json.filters = [] as VolJSON[];
        for (let i = 0; i < this.filters.length; i++) {
            json.filters.push(this.filters[i].toJSON());
        }
        json.biases = this.biases.toJSON() as VolJSON;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.num_inputs = json.num_inputs as number;
        this.l1_decay_mul = (typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0) as number;
        this.l2_decay_mul = (typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0) as number;
        this.filters = [];
        for (let i = 0; i < json.filters.length; i++) {
            const v = new Vol(0, 0, 0, 0);
            v.fromJSON(json.filters[i]);
            this.filters.push(v);
        }
        this.biases = new Vol(0, 0, 0, 0);
        this.biases.fromJSON(json.biases as Vol);
    }
}


import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

export interface PoolLayerOptions extends LayerOptions {
    /** <required> filter size */
    sx: number;
    /** <optional> filter size */
    sy?: number;
    /** <optional> */
    stride?: number;
    /** <optional> */
    pad?: number;
}

export class PoolLayer extends LayerBase implements ILayer {
    sx: number;
    sy: number;
    in_depth: number;
    in_sx: number;
    in_sy: number;
    stride: number;
    pad: number;
    switchx: number[] | Float64Array;
    switchy: number[] | Float64Array;
    in_act: Vol;
    out_act: Vol;

    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const popt = <PoolLayerOptions>opt;
        super(popt);

        // required
        this.sx = popt.sx; // filter size
        this.in_depth = popt.in_depth as number;
        this.in_sx = popt.in_sx as number;
        this.in_sy = popt.in_sy as number;

        // optional
        this.sy = typeof popt.sy !== 'undefined' ? popt.sy : this.sx;
        this.stride = typeof popt.stride !== 'undefined' ? popt.stride : 2;
        this.pad = typeof popt.pad !== 'undefined' ? popt.pad : 0; // amount of 0 padding to add around borders of input volume

        // computed
        this.out_depth = this.in_depth;
        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = 'pool';
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        this.switchx = util.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.switchy = util.zeros(this.out_sx * this.out_sy * this.out_depth);
    }

    forward(V: Vol, ) {
        this.in_act = V;

        const A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        let n = 0; // a counter for switches
        for (let d = 0; d < this.out_depth; d++) {
            let x = -this.pad;
            let y = -this.pad;
            for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                y = -this.pad;
                for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {

                    // convolve centered at this particular location
                    let a = -99999; // hopefully small enough ;\
                    let winx = -1, winy = -1;
                    for (let fx = 0; fx < this.sx; fx++) {
                        for (let fy = 0; fy < this.sy; fy++) {
                            const oy = y + fy;
                            const ox = x + fx;
                            if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                                const v = V.get(ox, oy, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if (v > a) { a = v; winx = ox; winy = oy; }
                            }
                        }
                    }
                    this.switchx[n] = winx;
                    this.switchy[n] = winy;
                    n++;
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return this.out_act;
    }
    backward() {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        const V = this.in_act;
        V.dw = util.zeros(V.w.length); // zero out gradient wrt data
        // const A = this.out_act; // computed in forward pass

        let n = 0;
        for (let d = 0; d < this.out_depth; d++) {
            let x = -this.pad;
            let y = -this.pad;
            for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                y = -this.pad;
                for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {

                    const chain_grad = this.out_act.get_grad(ax, ay, d);
                    V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
                    n++;

                }
            }
        }
    }

    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }

    toJSON() {
        const json: LayerJSON = {};
        json.sx = this.sx;
        json.sy = this.sy;
        json.stride = this.stride;
        json.in_depth = this.in_depth;
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.pad = this.pad;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.sx = json.sx as number;
        this.sy = json.sy as number;
        this.stride = json.stride as number;
        this.in_depth = json.in_depth as number;
        this.pad = (typeof json.pad !== 'undefined' ? json.pad : 0) as number; // backwards compatibility
        this.switchx = util.zeros(this.out_sx * this.out_sy * this.out_depth); // need to re-init these appropriately
        this.switchy = util.zeros(this.out_sx * this.out_sy * this.out_depth);
    }
}

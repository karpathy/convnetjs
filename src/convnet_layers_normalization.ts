import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

export interface LocalResponseNormalizationLayerOptions extends LayerOptions {
    /** <required> */
    k: number;
    /** <required> */
    n: number;
    /** <required> */
    alpha: number;
    /** <required> */
    beta: number;
}
/**
 * a bit experimental layer for now. I think it works but I'm not 100%
 * the gradient check is a bit funky. I'll look into this a bit later.
 * Local Response Normalization in window, along depths of volumes
 */
export class LocalResponseNormalizationLayer extends LayerBase implements ILayer {
    k: number;
    n: number;
    alpha: number;
    beta: number;
    in_act: Vol;
    out_act: Vol;
    S_cache_: Vol;


    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const lrnopt = <LocalResponseNormalizationLayerOptions>opt;
        super(lrnopt);

        // required
        this.k = lrnopt.k;
        this.n = lrnopt.n;
        this.alpha = lrnopt.alpha;
        this.beta = lrnopt.beta;

        // computed
        this.out_sx = lrnopt.in_sx as number;
        this.out_sy = lrnopt.in_sy as number;
        this.out_depth = lrnopt.in_depth as number;
        this.layer_type = 'lrn';

        // checks
        if (this.n % 2 === 0) { console.log('WARNING n should be odd for LRN layer'); }
    }
    forward(V: Vol, ) {
        this.in_act = V;

        const A = V.cloneAndZero();
        this.S_cache_ = V.cloneAndZero();
        const n2 = Math.floor(this.n / 2);
        for (let x = 0; x < V.sx; x++) {
            for (let y = 0; y < V.sy; y++) {
                for (let i = 0; i < V.depth; i++) {

                    const ai = V.get(x, y, i);

                    // normalize in a window of size n
                    let den = 0.0;
                    for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                        const aa = V.get(x, y, j);
                        den += aa * aa;
                    }
                    den *= this.alpha / this.n;
                    den += this.k;
                    this.S_cache_.set(x, y, i, den); // will be useful for backprop
                    den = Math.pow(den, this.beta);
                    A.set(x, y, i, ai / den);
                }
            }
        }

        this.out_act = A;
        return this.out_act; // dummy identity function for now
    }
    backward() {
        // evaluate gradient wrt data
        const V = this.in_act; // we need to set dw of this
        V.dw = util.zeros(V.w.length); // zero out gradient wrt data
        // let A = this.out_act; // computed in forward pass

        const n2 = Math.floor(this.n / 2);
        for (let x = 0; x < V.sx; x++) {
            for (let y = 0; y < V.sy; y++) {
                for (let i = 0; i < V.depth; i++) {

                    const chain_grad = this.out_act.get_grad(x, y, i);
                    const S = this.S_cache_.get(x, y, i);
                    const SB = Math.pow(S, this.beta);
                    const SB2 = SB * SB;

                    // normalize in a window of size n
                    for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                        const aj = V.get(x, y, j);
                        let g = -aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha / this.n * 2 * aj;
                        if (j === i) { g += SB; }
                        g /= SB2;
                        g *= chain_grad;
                        V.add_grad(x, y, j, g);
                    }

                }
            }
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }

    toJSON() {
        const json: LayerJSON = {};
        json.k = this.k;
        json.n = this.n;
        json.alpha = this.alpha; // normalize by size
        json.beta = this.beta;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.out_depth = this.out_depth;
        json.layer_type = this.layer_type;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.k = json.k as number;
        this.n = json.n as number;
        this.alpha = json.alpha as number; // normalize by size
        this.beta = json.beta as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.out_depth = json.out_depth as number;
        this.layer_type = json.layer_type as string;
    }
}

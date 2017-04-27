import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

export interface DorpoutLayerOptions extends LayerOptions {
    /** <required> */
    drop_prob: number;
}

/**
 * An inefficient dropout layer
 * Note this is not most efficient implementation since the layer before
 * computed all these activations and now we're just going to drop them :(
 * same goes for backward pass. Also, if we wanted to be efficient at test time
 * we could equivalently be clever and upscale during train and copy pointers during test
 * todo: make more efficient.
 */
export class DropoutLayer extends LayerBase implements ILayer {
    in_act: Vol;
    drop_prob: number;
    dropped: boolean[];
    out_act: Vol;

    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        const dopt = <DorpoutLayerOptions>opt;
        super(dopt);

        // computed
        this.out_sx = dopt.in_sx as number;
        this.out_sy = dopt.in_sy as number;
        this.out_depth = dopt.in_depth as number;
        this.layer_type = 'dropout';
        this.drop_prob = typeof dopt.drop_prob !== 'undefined' ? dopt.drop_prob : 0.5;
        const d = <number[]>util.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.dropped = d.map((v) => v !== 0);
    }
    forward(V: Vol, is_training?: boolean) {
        this.in_act = V;
        if (typeof (is_training) === 'undefined') { is_training = false; } // default is prediction mode
        const V2 = V.clone();
        const N = V.w.length;
        if (is_training) {
            // do dropout
            for (let i = 0; i < N; i++) {
                if (Math.random() < this.drop_prob) { V2.w[i] = 0; this.dropped[i] = true; } // drop!
                else { this.dropped[i] = false; }
            }
        } else {
            // scale the activations during prediction
            for (let i = 0; i < N; i++) { V2.w[i] *= this.drop_prob; }
        }
        this.out_act = V2;
        return this.out_act; // dummy identity function for now
    }
    backward() {
        const V = this.in_act; // we need to set dw of this
        const chain_grad = this.out_act;
        const n = V.w.length;
        V.dw = util.zeros(n); // zero out gradient wrt data
        for (let i = 0; i < n; i++) {
            if (!(this.dropped[i])) {
                V.dw[i] = chain_grad.dw[i]; // copy over the gradient
            }
        }
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
        json.drop_prob = this.drop_prob;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
        this.drop_prob = json.drop_prob as number;

        const d = <number[]>util.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.dropped = d.map((v) => v !== 0);
    }
}

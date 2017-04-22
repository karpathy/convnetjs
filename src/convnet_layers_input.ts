import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptions, ILayer, LayerJSON, ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";
const getopt = util.getopt;

export interface InputLayerOptions extends LayerOptions {
    [key: string]: number | string;
    out_depth: number;
    depth: number;
    width: number;
    height: number;
    out_sx: number;
    out_sy: number;
    sx: number;
    sy: number;
}

export class InputLayer extends LayerBase implements ILayer {
    out_depth: number;
    out_sx: number;
    out_sy: number;
    in_act: Vol;
    out_act: Vol;

    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
        super(opt);

        // required: depth
        this.out_depth = getopt(opt, ['out_depth', 'depth'], 0);

        // optional: default these dimensions to 1
        this.out_sx = getopt(opt, ['out_sx', 'sx', 'width'], 1);
        this.out_sy = getopt(opt, ['out_sy', 'sy', 'height'], 1);

        // computed
        this.layer_type = 'input';
    }
    forward(V: Vol, ) {
        this.in_act = V;
        this.out_act = V;
        return this.out_act; // simply identity function for now
    }
    backward() { }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): LayerJSON {
        const json: LayerJSON = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        return json;
    }
    fromJSON(json: LayerJSON) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as string;
    }
}

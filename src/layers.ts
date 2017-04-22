import { Vol, VolJSON } from "./convnet_vol";
export interface LayerOptions {
    [key: string]: number | string;
    type: string;
}

export interface LayerJSON {
    [key: string]: number | string | number[] | VolJSON | VolJSON[];
    filters?: VolJSON[];
}

export interface ParamsAndGrads {
    params: number[] | Float64Array;
    grads: number[] | Float64Array;
    l2_decay_mul: number;
    l1_decay_mul: number;
}


export interface ILayer {
    layer_type: string;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    forward(V: Vol, is_training: boolean): Vol;
    backward(y?: number | number[] | Float64Array | { [key: string]: number }): void | number;
    getParamsAndGrads(): ParamsAndGrads[];
    toJSON(): LayerJSON;
    fromJSON(json: LayerJSON): void;
}

export class LayerBase {
    layer_type: string;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    constructor(opt?: LayerOptions) {
        if (!opt) { return; }
    }
}

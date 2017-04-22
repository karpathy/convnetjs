import { Vol } from "./convnet_vol";
export interface LayerOptions {
    [key: string]: number | string;
    type: string;
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
    getParamsAndGrads(): any[];
    toJSON(): any;
    fromJSON(json: any): void;
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

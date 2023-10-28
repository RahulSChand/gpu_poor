import React, { useState, useEffect, useRef } from "react";
import TextInput from "./textBox";
import Modal from "react-modal";
import fullText from "./textContent";
import gpuJSONData from "./gpu_config.json";
import cpuJSONData from "./cpu_config.json";

const billion = 1000000000;
const tera = 1000000000 * 1000;
let configPath = "/gpu_poor/all_configs.json";
if (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
) {
    configPath = "/gpu_poor/all_configs.json";
}
const MAX_FILE_SIZE = 500000;
const ggml_quants = [
    "ggml_QK4_0",
    "ggml_QK4_1",
    "ggml_QK5_0",
    "ggml_QK5_1",
    "ggml_QK8_0",
    "ggml_QK8_1",

    "ggml_Q2_K",

    "ggml_Q3_K_L",
    "ggml_Q3_K_M",

    "ggml_QK4_K_M",
    "ggml_QK4_K_S",

    "ggml_QK5_K_M",
    "ggml_Q6_K",
];
// console.log(configPath);

/*
dropdownTrnOrNot: 'inf', 'trn', 'inf_vLLM','inf_exL','inf_ggml'
dropdownFullOrNot: 'lora_trn, 'full_trn', 'qlora'
dropdownOpt: 'no_opt', 'sgd_opt','adam_opt'
dropdownQuant: 'no_quant','bnb_int8','bnb_q4', 
*/
const specialNamesMapping = {
    "meta-llama/Llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-13-7b": "meta-llama/Llama-13-7b-hf",
    "meta-llama/Llama-2-70b": "meta-llama/Llama-13-70b-hf",
};

function specialMapping(name) {
    if (name in specialNamesMapping) {
        return specialNamesMapping[name];
    }
    return name;
}

function getKey(keys, obj, defaultVal) {
    let toReturn = null;
    for (const key of keys) {
        if (obj.hasOwnProperty(key)) {
            // console.log("found: ",key);
            toReturn = obj[key];
            break;
        }
    }
    if (toReturn == null) {
        return defaultVal;
    }
    return toReturn;
}

function computeOverheadGGML(contextLen) {
    return 0.1 * contextLen;
}

function computeInferenceOnlyActivationMemory(contextLen, parsedConfig) {
    const hiddenDim = parsedConfig["hiddenDim"];
    const heads = parsedConfig["heads"];

    //return ((1000*4096*5)*2 + (1000*1000*32*2))/(1024*1024)
    return (
        (contextLen * hiddenDim * 5 * 2 + contextLen * contextLen * heads * 2) /
        (1024 * 1024)
    );
}

//floatBytes, quant
function computeModelSizeGGML(parsedConfig, quant) {
    const vocab = parsedConfig["vocab"],
        heads = parsedConfig["heads"],
        numLayers = parsedConfig["num_layers"],
        hiddenDim = parsedConfig["hiddenDim"],
        interDim = parsedConfig["interDim"];

    const totalParams =
        vocab * hiddenDim * 2 +
        numLayers * 4 * hiddenDim * hiddenDim +
        numLayers * 3 * interDim * hiddenDim;

    const other_v_down_params =
        numLayers * hiddenDim * hiddenDim + numLayers * hiddenDim * interDim;

    const other_params_Q2K =
        totalParams -
        (hiddenDim * hiddenDim * numLayers * 2 + 2 * vocab * hiddenDim);

    const mult_factor_dic = {
        ggml_QK4_0: 18,
        ggml_QK4_1: 20,
        ggml_QK5_0: 22,
        ggml_QK5_1: 24,
        ggml_QK8_0: 34,
        ggml_QK8_1: 40,
    };

    const mult_factor_dic_64 = {
        ggml_Q6_K: 54.0,
        ggml_Q3: 26.0,
        ggml_Q4: 38.0,
        ggml_Q5: 46.0,
    };

    //Q2_K is 22.0

    const mult_factor_dic_combination = {
        ggml_Q3_K_L: [38.0, 26.0],
        ggml_Q3_K_M: [46.0, 26.0],
        ggml_QK4_K_S: [46.0, 38.0],
        ggml_QK4_K_M: [54.0, 38.0],
        ggml_QK5_K_M: [54.0, 46.0],
        ggml_Q2_K: [26.0, 22.0],
    };

    let total = 0;
    if (mult_factor_dic.hasOwnProperty(quant)) {
        total = (mult_factor_dic[quant] * totalParams) / (32 * 1024 * 1024);
    }
    if (mult_factor_dic_64.hasOwnProperty(quant)) {
        total = (mult_factor_dic_64[quant] * totalParams) / (64 * 1024 * 1024);
    }
    if (mult_factor_dic_combination.hasOwnProperty(quant)) {
        const factors = mult_factor_dic_combination[quant];

        if (quant === "ggml_Q2_K") {
            total =
                ((totalParams - other_params_Q2K) * factors[1] +
                    other_params_Q2K * factors[0]) /
                (64 * 1024 * 1024);
        } else {
            total =
                ((totalParams - other_v_down_params) * factors[1] +
                    other_v_down_params * factors[0]) /
                (64 * 1024 * 1024);
        }
    }

    return total;
}

function computeModelSize(parsedConfig) {
    const vocab = parsedConfig["vocab"],
        heads = parsedConfig["heads"],
        numLayers = parsedConfig["num_layers"],
        hiddenDim = parsedConfig["hiddenDim"],
        interDim = parsedConfig["interDim"];

    // console.log(vocab, heads, numLayers, hiddenDim, interDim);
    // let fB = floatBytes;
    // if (quant === 'bnb_int8'){fB = 1;}
    // if (quant === 'bnb_q4'){fB = 0.5;}

    const out =
        vocab * hiddenDim * 2 +
        numLayers * 4 * hiddenDim * hiddenDim +
        numLayers * 3 * interDim * hiddenDim;
    // console.log("this is out: ", out)

    return out;
}

function getGradOptMemory(
    dropdownFullOrNot,
    dropdownOpt,
    dropdownQuant,
    modelSize,
    floatBytes,
    parsedConfig,
    contextLen,
    batchSize = 1
) {
    const full = dropdownFullOrNot,
        opt = dropdownOpt,
        quant = dropdownQuant;
    console.log(full, opt, quant);

    //QLora start
    // console.log("full: ", full);
    if (full === "qlora" && opt === "adam_opt") {
        //Need to check if q4 also takes extra memory
        console.log("calculating qlora");
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 0.5 * 4 * 3 +
            getExtraMemory(parsedConfig, "qlora", contextLen) * batchSize
        );
    }
    if (full === "qlora" && opt === "sgd_opt") {
        //Need to check if q4 also takes extra memory
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 0.5 * 4 * 1 +
            getExtraMemory(parsedConfig, "qlora", contextLen) * batchSize
        );
    }
    //QLora end

    if (full === "full_trn" && opt === "adam_opt" && quant === "no_quant") {
        return modelSize * 3 * floatBytes;
    }

    if (full === "full_trn" && opt === "adam_opt" && quant === "bnb_int8") {
        return (
            modelSize * 3 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        ); //Some extra mmeory that bnb int8 takes
    }

    if (full === "full_trn" && opt === "adam_opt" && quant === "bnb_q4") {
        //Need to check if q4 also takes extra memory
        return (
            modelSize * 3 * 0.5 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    //------------
    if (full === "full_trn" && opt === "sgd_opt" && quant === "no_quant") {
        return modelSize * 1 * floatBytes;
    }

    if (full === "full_trn" && opt === "sgd_opt" && quant === "bnb_int8") {
        return (
            modelSize * 1 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    if (full === "full_trn" && opt === "sgd_opt" && quant === "bnb_q4") {
        return (
            modelSize * 1 * 0.5 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    //4*layer*8*hid*4*2

    //------------
    if (full === "lora_trn" && opt === "adam_opt" && quant === "no_quant") {
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 * 2
        );
    }

    if (full === "lora_trn" && opt === "adam_opt" && quant === "bnb_int8") {
        
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    if (full === "lora_trn" && opt === "adam_opt" && quant === "bnb_q4") {
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 3 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    //------------
    if (full === "lora_trn" && opt === "sgd_opt" && quant === "no_quant") {
        return parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 2;
    }

    if (full === "lora_trn" && opt === "sgd_opt" && quant === "bnb_int8") {
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    if (full === "lora_trn" && opt === "sgd_opt" && quant === "bnb_q4") {
        return (
            parsedConfig.num_layers * 8 * parsedConfig.hiddenDim * 2 * 4 * 1 +
            getExtraMemory(parsedConfig, quant, contextLen) * batchSize
        );
    }

    console.log(full, opt, quant);
    throw new Error("Invalid combination of values");
}

function getExtraMemory(parsedConfig, quant, contextLen) {
    const constant_8_extra = 0.75;
    const constant_4_extra = 1.0;
    const constant_qlora = 0.75;

    const common =
        (10 * parsedConfig.hiddenDim +
            5 * parsedConfig.hiddenDim +
            4 * parsedConfig.interDim +
            2 * parsedConfig.interDim) *
        parsedConfig.num_layers;

    let extra_mem = 0;
    let contextLenSqrtRoot = 1.0;
    // if (contextLen > 100){
    //     contextLenSqrtRoot = Math.round(Math.sqrt(contextLen));
    // }
    // else{
    //     contextLenSqrtRoot = contextLen;
    // }
    const baseLen = 50;
    const ratioContextLen = contextLen / 50;
    if (ratioContextLen > 1.0) {
        contextLenSqrtRoot = Math.sqrt(ratioContextLen);
    }

    if (quant === "bnb_int8") {
        extra_mem =
            constant_8_extra * common * baseLen * contextLenSqrtRoot * 1.25;
    }

    if (quant === "bnb_q4") {
        extra_mem =
            constant_4_extra * common * baseLen * contextLenSqrtRoot * 1.0;
    }

    if (quant === "qlora") {
        extra_mem =
            constant_qlora * common * baseLen * contextLenSqrtRoot * 1.0;
    }

    console.log("extra mem", extra_mem);
    return extra_mem;
}

function getExtraMemoryOld(parsedConfig, quant) {
    const constant_8_overhead = 200.0,
        constant_8_extra = 350.0;
    const constant_4_overhead = 350.0,
        constant_4_extra = 550.0;

    const common =
        (10 * parsedConfig.hiddenDim +
            5 * parsedConfig.hiddenDim +
            4 * parsedConfig.interDim +
            2 * parsedConfig.interDim) *
        parsedConfig.num_layers;

    let extra_mem = 0;

    if (quant === "bnb_int8") {
        extra_mem = constant_8_overhead * common + constant_8_extra * common;
    }

    if (quant === "bnb_q4") {
        extra_mem = constant_4_overhead * common + constant_4_extra * common;
    }

    console.log("extra mem", extra_mem);
    return extra_mem;
}

function getActivationMemory(
    parsedConfig,
    contextLen,
    floatBytes,
    quant,
    dropdownFullOrNot,
    batchSize = 1
) {
    const heads = parsedConfig["heads"],
        numLayers = parsedConfig["num_layers"],
        hiddenDim = parsedConfig["hiddenDim"],
        interDim = parsedConfig["interDim"];

    let fB = floatBytes;
    const len = contextLen;

    // if (quant==='bnb_int8'){fB=1;}
    // if (quant==='bnb_q4'){fB=0.5;}

    console.log("activation: ", heads, numLayers, hiddenDim, interDim);

    //const attn_per_layer = qkv + qk (transpose) + attn mat + attn mat convert tp fp32 + attn  mat divided by sqrt +
    const attn_per_layer =
        len * hiddenDim * 3 * fB +
        len * hiddenDim * 2 * fB +
        len * len * heads * fB +
        len * len * heads * 4 +
        len * len * heads * fB +
        len * hiddenDim * fB +
        len * hiddenDim * fB +
        len * hiddenDim * fB;

    // heads*len*len*4 + heads*len*len*fB + 3*hiddenDim*len*fB + hiddenDim*len*fB + hiddenDim*len*fB

    const ffn_per_layer =
        hiddenDim * len * fB +
        hiddenDim * len * fB +
        fB * 5 * len * interDim +
        interDim * len * fB;

    const norm = len * 4 * 2 + len * hiddenDim * fB * 6;

    let lora = 0;
    // if (dropdownFullOrNot==='lora_trn'){
    //   lora = (8*len*2 + hiddenDim*len*2)*4;
    // }

    const total_per_layer = attn_per_layer + ffn_per_layer + norm + lora;
    console.log(
        "total per layer: ",
        convertToMB(attn_per_layer),
        convertToMB(ffn_per_layer),
        convertToMB(norm),
        convertToMB(lora)
    );

    //total per layer: 4.2724609375 5.55419921875 6.409454345703125 8.02001953125
    let total = total_per_layer * numLayers;
    total = total * batchSize;

    console.log("this is total: ", total, attn_per_layer + ffn_per_layer);

    return total;
}

function checkCombinationTrainInference(
    quantType,
    setErrorMessage,
    openModal,
    typeOfTrn
) {
    //! Can't train full with QLoRA
    if (typeOfTrn === "full_trn" && ggml_quants.includes(quantType)) {
        setErrorMessage("Can't use GGML for training");
        openModal();
        return false;
    }
    if (typeOfTrn === "qlora" && quantType != "no_quant") {
        setErrorMessage(
            "QLoRA is 4bit explicit. No need to select a quant type if you are training using QLoRA. Set it to 'None'"
        );
        openModal();
        return false;
    }
    return true;
}

function checkCombinationInferenceTok(
    trnType,
    quantType,
    setErrorMessage,
    openModal
) {
    if (ggml_quants.includes(quantType)) {
        if (trnType != "inf_ggml") {
            setErrorMessage(
                "Invalid combination of inference type/quantization"
            );
            openModal();
            return false;
        }
    }
    if (quantType != "no_quant" && trnType === "inf_vLLM") {
        setErrorMessage("vLLm doesn't support quant (maybe)");
        openModal();
        return false;
    }
    if (
        trnType === "inf_ggml" &&
        (quantType === "bnb_int8" || quantType === "bnb_q4")
    ) {
        setErrorMessage("ggml doesn't support bnb");
        openModal();
        return false;
    }


    return true;
}

function checkCombinationInference(
    trnType,
    quantType,
    setErrorMessage,
    openModal
) {
    if (ggml_quants.includes(quantType)) {
        if (trnType != "inf_ggml") {
            setErrorMessage(
                "Invalid combination of inference type/quantization"
            );
            openModal();
            return false;
        }
    }
    if (quantType != "no_quant" && trnType === "inf_vLLM") {
        setErrorMessage("vLLm doesn't support quant (maybe)");
        openModal();
        return false;
    }
    if (
        trnType === "inf_ggml" &&
        (quantType === "bnb_int8" || quantType === "bnb_q4")
    ) {
        setErrorMessage("ggml doesn't support bnb");
        openModal();
        return false;
    }
    if (trnType === "inf_ggml" && quantType === "no_quant") {
        setErrorMessage(
            "If you want no quant then pick vLLM/HF inference framework"
        );
        openModal();
        return false;
    }

    if (trnType === "inf_exL") {
        setErrorMessage("exLlama hasn't been added yet :)");
        openModal();
        return false;
    }
    return true;
}

function sanityUploadedConfig(jsonUploadedData, setErrorMessage, openModal) {
    function uploadError() {
        setErrorMessage(
            "upload config doesn't have correct keys. make sure your config has the keys present in https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json"
        );
        openModal();
        return null;
    }

    if (Object.keys(jsonUploadedData).length === 0) {
        setErrorMessage("Uploaded json is empty :)");
        openModal();
        return null; // JSON is empty
    }

    // console.log(jsonUploadedData);

    let vocab = 0,
        hiddenDim = 0,
        heads = 0,
        interDim = 0,
        num_layers = 0;

    if (jsonUploadedData.hasOwnProperty("vocab_size")) {
        vocab = jsonUploadedData["vocab_size"];
    } else {
        uploadError();
        return null;
    }

    if (jsonUploadedData.hasOwnProperty("hidden_size")) {
        hiddenDim = jsonUploadedData["hidden_size"];
    } else {
        uploadError();
        return null;
    }

    if (jsonUploadedData.hasOwnProperty("num_attention_heads")) {
        heads = jsonUploadedData["num_attention_heads"];
    } else {
        uploadError();
        return null;
    }

    if (jsonUploadedData.hasOwnProperty("intermediate_size")) {
        interDim = jsonUploadedData["intermediate_size"];
    } else {
        uploadError();
        return null;
    }

    if (jsonUploadedData.hasOwnProperty("num_hidden_layers")) {
        num_layers = jsonUploadedData["num_hidden_layers"];
    } else {
        uploadError();
        return null;
    }

    return {
        vocab: vocab,
        hiddenDim: hiddenDim,
        heads: heads,
        interDim: interDim,
        num_layers: num_layers,
    };
}

function getParseConfig(parsedJSONData, setErrorMessage, openModal) {
    console.log(Object.keys(parsedJSONData).length);
    if (Object.keys(parsedJSONData).length == 0) {
        setErrorMessage(
            "Huggingface config of this id doesn't have correct keys. e.g. this is a ggml model. Please upload your config in correct format"
        );
        openModal();
        return null;
    }

    const vocab = getKey(["vocab_size"], parsedJSONData, 32000);
    const hiddenDim = getKey(
        ["hidden_size", "d_model", "n_embd"],
        parsedJSONData,
        768
    );
    const heads = getKey(
        ["num_attention_heads", "num_heads", "n_head"],
        parsedJSONData,
        12
    );
    const interDim = getKey(
        ["intermediate_size", "n_inner", "d_ff"],
        parsedJSONData,
        hiddenDim * 4
    );
    const num_layers = getKey(
        ["num_layers", "num_hidden_layers", "n_layer"],
        parsedJSONData,
        12
    );

    return {
        vocab: vocab,
        hiddenDim: hiddenDim,
        heads: heads,
        interDim: interDim,
        num_layers: num_layers,
    };
}

function getDefault(modelSize) {
    //If only model size is provided. Guess the values
    let vocab = null;
    let heads = null;
    let numLayers = null;

    function getApprox(modelSize) {
        let vocabR = null,
            headsR = null,
            numLayersR = null;
        if (modelSize < 5) {
            vocabR = 32000;
            headsR = 32;
            numLayersR = 24;
            return [vocabR, headsR, numLayersR];
        }
        if (modelSize < 10) {
            vocabR = 32000;
            headsR = 32;
            numLayersR = 32;
            return [vocabR, headsR, numLayersR];
        }
        if (modelSize < 24) {
            vocabR = 32000;
            headsR = 40;
            numLayersR = 40;
            return [vocabR, headsR, numLayersR];
        }

        if (modelSize < 55) {
            vocabR = 32000;
            headsR = 64;
            numLayersR = 48;
            return [vocabR, headsR, numLayersR];
        }

        vocabR = 32000;
        headsR = 64;
        numLayersR = 80;
        return [vocabR, headsR, numLayersR];
    }

    [vocab, heads, numLayers] = getApprox(modelSize);

    //vocab*h + numLayers*4*h*h + 3*4*h*h*numLayers = modelSize*10^9
    const A = numLayers * 4 + 3 * 4 * numLayers;
    const B = 2 * vocab;
    const C = -1 * modelSize * billion;

    let h = (-B + Math.sqrt(B * B - 4 * A * C)) / (2 * A);
    h = Math.ceil(h);

    return {
        vocab: vocab,
        hiddenDim: h,
        heads: heads,
        interDim: 4 * h,
        num_layers: numLayers,
    };
}

function convertToMB(value) {
    return value / (1024 * 1024);
}

function convertToMBModelSize(value, quant, typeOfTrn) {
    let extra = 0;
    let fB = 2;
    let size = (value * fB) / (1024 * 1024);
    if (quant === "bnb_int8" || quant === "bnb_q4" || typeOfTrn === "qlora") {
        extra = 0.06 * size;
    }

    if (quant === "bnb_int8") {
        size = size / 2;
    }
    if (quant === "bnb_q4") {
        size = size / 4;
    }

    if (typeOfTrn === "qlora") {
        size = size / 4 - (value * 2) / (64 * 1024 * 1024);
    }

    return size + extra;
}

function convertToBytes(floatType) {
    return 2.0;
}

function getAllComputedData(
    parsedJSONData,
    jsonUploadedData,
    modelSize,
    contextLen,
    floatType,
    selections,
    setErrorMessage,
    openModal,
    batchSize
) {
    let parsedConfig = null,
        modelSizeinB = null;
    let activationMemory = 0,
        gradAndOptMemory = 0;
    let inferenceMemory = 0;
    let totalMemory = 0;
    const floatBytes = convertToBytes(floatType);
    const quantType = selections.dropdownQuant;
    const trnType = selections.dropdownTrnOrNot;
    const typeOfTrn = selections.dropdownFullOrNot;

    //trnType should be trnOrNot

    if (batchSize === "") {
        batchSize = "1";
    }

    let overHead = 650;
    if (!isValidPositiveInteger(contextLen)) {
        setErrorMessage(
            "Context len can't be blank or have non numeric or negative/zero values."
        );
        openModal();
        return null;
    }

    if (!isValidPositiveInteger(batchSize)) {
        setErrorMessage(
            "Batch size cant have non numeric or negative/zero values"
        );
        openModal();
        return null;
    }

    if (parsedJSONData == null) {
        if (jsonUploadedData != null) {
            parsedConfig = sanityUploadedConfig(
                jsonUploadedData,
                setErrorMessage,
                openModal
            );
            console.log(parsedConfig, "uploaded");
            if (parsedConfig == null) {
                return null;
            }
            modelSizeinB = computeModelSize(parsedConfig);
        } else {
            if (!isNumberOrFloat(modelSize)) {
                console.log("error with model size");
                setErrorMessage(
                    "Hugginface model id not available, enter model size(>0) or upload config"
                );
                openModal();
                return null;
            }

            parsedConfig = getDefault(modelSize);
            modelSizeinB = modelSize * billion;
        }
    } else {
        parsedConfig = getParseConfig(
            parsedJSONData,
            setErrorMessage,
            openModal
        );
        if (parsedConfig == null) {
            return null;
        }
        console.log(parsedConfig);
        modelSizeinB = computeModelSize(parsedConfig);
    }

    let fB = floatBytes;
    if (quantType === "bnb_int8") {
        fB = 1;
    }
    if (quantType === "bnb_q4" || typeOfTrn === "qlora") {
        fB = 0.5;
    }
    let modelSizeinMB = convertToMBModelSize(
        modelSizeinB,
        quantType,
        typeOfTrn
    );
    // console.log(modelSizeinB);

    //!Inference
    if (trnType != "trn") {
        let checkSanity = checkCombinationInference(
            trnType,
            quantType,
            setErrorMessage,
            openModal
        );
        if (!checkSanity) {
            return null;
        }

        if (trnType === "inf" || trnType === "inf_vLLM") {
            let fB = 2;
            //If bnb quant
            if (quantType === "bnb_int8") {
                fB = 1;
            }
            if (quantType === "bnb_q4" || typeOfTrn === "qlora") {
                fB = 0.5;
            }

            inferenceMemory = convertToMB(
                2 *
                    contextLen *
                    2 *
                    2 *
                    parsedConfig["hiddenDim"] *
                    parsedConfig["num_layers"]
            );

            activationMemory = computeInferenceOnlyActivationMemory(
                contextLen,
                parsedConfig
            );

            console.log(
                "HERE!!!",
                inferenceMemory,
                modelSizeinMB,
                overHead,
                activationMemory
            );
        }
        if (trnType === "inf_ggml") {
            modelSizeinMB = computeModelSizeGGML(parsedConfig, quantType);
            inferenceMemory = convertToMB(
                1 *
                    contextLen *
                    2 *
                    2 *
                    parsedConfig["hiddenDim"] *
                    parsedConfig["num_layers"]
            );
            activationMemory = computeInferenceOnlyActivationMemory(
                contextLen,
                parsedConfig
            );
            overHead = overHead + computeOverheadGGML(contextLen);
        }

        totalMemory =
            inferenceMemory + modelSizeinMB + overHead + activationMemory;
    } else {
        // console.log("training!");

        let checkSanity = checkCombinationTrainInference(
            quantType,
            setErrorMessage,
            openModal,
            typeOfTrn
        );
        if (!checkSanity) {
            return null;
        }
        //! Train
        activationMemory = getActivationMemory(
            parsedConfig,
            contextLen,
            floatBytes,
            quantType,
            typeOfTrn,
            batchSize
        );

        activationMemory = convertToMB(activationMemory);
        // console.log("got activation", activationMemory);

        gradAndOptMemory = getGradOptMemory(
            typeOfTrn,
            selections.dropdownOpt,
            quantType,
            modelSizeinB,
            floatBytes,
            parsedConfig,
            contextLen,
            batchSize
        );

        // console.log("got gradOpt", gradAndOptMemory);

        gradAndOptMemory = convertToMB(gradAndOptMemory);
        totalMemory = modelSizeinMB + gradAndOptMemory + activationMemory;

        console.log("got total", totalMemory);

        totalMemory = totalMemory + overHead;
    }

    return {
        Total: Math.ceil(totalMemory),
        "KV Cache": Math.ceil(inferenceMemory),
        "Model Size": Math.ceil(modelSizeinMB),
        "Activation Memory": Math.ceil(activationMemory),
        "Grad & Optimizer memory": Math.ceil(gradAndOptMemory),
        "cuda + other overhead": overHead,
    };
}

///Users/rahulchand/gpu_mem/public/all_configs.json
async function fetchParams(name) {
    // let output = fetch('https://huggingface.co/meta-llama/Llama-2-7b/raw/main/params.json');

    let response = await fetch(configPath);
    response = await response.json();
    // console.log(response.hasOwnProperty(name));

    return response.hasOwnProperty(name) ? response[name] : null;
}

// function isNumberOrFloat(value) {
//     return /^-?\d+(\.\d+)?$/.test(value);
// }

function isNumberOrFloat(value) {
    const num = parseFloat(value);
    return !isNaN(num) && num > 0;
}

function isValidPositiveInteger(input) {
    const num = parseFloat(input);
    console.log(num, input);
    // console.log("isvalid :", input);

    return Number.isInteger(num) && num > 0;
}

function getGPUDataFromJSON() {}

function App() {
    // let subtitle;
    const [modelSize, setModelSize] = useState("");
    const [modelName, setModelName] = useState("");
    const [contextLen, setContextLen] = useState("");

    const [promptLen, setPromptLen] = useState("");

    const [batchSize, setBatchSize] = useState(1);
    const [totalMemoryShown, setTotalMemoryShown] = useState(0);

    const [gpuJsonDataForTable, setGPUJSONDataForTable] = useState([]);
    const [cpuJsonDataForTable, setCPUJSONDataForTable] = useState([]);

    // const [breakDownMemory, setBreakDownMemory] = useState(" ");

    const [breakDownMemoryJson, setBreakDownMemoryJson] = useState([]);

    const [errorMessage, setErrorMessage] = useState("");

    const [fileNameUpload, setFileNameUpload] = useState("");

    const [modalIsOpen, setIsOpen] = React.useState(false);

    const [responseCache, setResponseCache] = useState(null);
    const [responseCacheKeys, setResponseCacheKeys] = useState(null);

    const [suggestions, setSuggestions] = useState([]);
    const [selectedIdx, setSelectedIdx] = useState(-1);
    const [tokenPerSecond, setTokenPerSecond] = useState("");

    const [numGPU, setNumGPU] = useState(1);
    const [numGPUINeed, setNumGPUINeed] = useState(null);
    const [memReqHardwareName, setMemReqHardwareName] = useState("");
    const [compReqHardwareName, setCompReqHardwareName] = useState("");

    const [compReqHardwareNameBefore, setCompReqHardwareNameBefore] =
        useState("");

    const [numOffload, setNumOffLoad] = useState(1);

    const [computedTokenPerSecond, setComputedTokenPerSecond] = useState(1);

    const [jsonData, setJsonData] = useState(null);

    const [jsonDataCompute, setJsonDataCompute] = useState(null);

    const [showSuggestions, setShowSuggestions] = useState(true);
    const [showDDR, setShowDDR] = useState([1, 0]);

    const [showTable, setShowTable] = useState(false);
    const [showTableGPU, setShowTableGPU] = useState(false);
    const [showTableCPU, setShowTableCPU] = useState(false);
    const [showTableCompute, setShowTableCompute] = useState(false);
    const [showTableComputeDisclaimer, setShowTableComputeDisclaimer] =
        useState("");
    const [showTableComputeSmallInfo, setShowTableComputeSmallInfo] =
        useState(0);

    const [faqOpen, setFaqOpen] = useState(false);

    // const th_css = "py-2 px-4 border bg-gray-200 text-gray-600 ";

    // const jsonDataSample = [
    //     { index: 1, name: "Alice", value: 30 },
    //     { index: 2, name: "Bob", value: 40 },
    //     { index: 3, name: "Carol", value: 50 },
    // ];

    function openModal() {
        setIsOpen(true);
    }

    function closeModal() {
        setIsOpen(false);
    }

    const handleFileClear = (event) => {
        setFileNameUpload("");
        setJsonData(null);
        // setTotalMemoryShown("");
        // setBreakDownMemory("");
    };

    const [displayedText, setDisplayedText] = useState("");
    const [isVisible, setIsVisible] = useState(true);
    const intervalIdRef = useRef(null);
    const wordIndexRef = useRef(0);
    const timeoutIdRef = useRef(null);

    const handleClickGenerateText = () => {
        let token_per_second = parseInt(tokenPerSecond, 10);

        setIsVisible(true);
        const words = fullText.split(/[\s,.;!?]+/);
        // console.log(words);
        wordIndexRef.current = 0; // reset word index
        setDisplayedText("");

        // Clear any existing interval before setting up a new one
        if (intervalIdRef.current) {
            clearInterval(intervalIdRef.current);
        }
        if (timeoutIdRef.current) {
            clearTimeout(timeoutIdRef.current);
        }

        intervalIdRef.current = setInterval(() => {
            if (wordIndexRef.current < words.length - 1) {
                wordIndexRef.current++;
                setDisplayedText((prevText) => {
                    if (prevText) {
                        return prevText + " " + words[wordIndexRef.current];
                    }
                    return words[wordIndexRef.current]; // No preceding space for the first word
                });
            }
        }, 1000 / token_per_second);
    };

    const handleClearGeneratedText = () => {
        if (intervalIdRef.current) {
            clearInterval(intervalIdRef.current);
        }
        if (timeoutIdRef.current) {
            clearTimeout(timeoutIdRef.current);
        }
        setDisplayedText("");
        setIsVisible(false);
    };

    useEffect(() => {
        return () => {
            if (intervalIdRef.current) {
                clearInterval(intervalIdRef.current);
            }
            if (timeoutIdRef.current) {
                clearTimeout(timeoutIdRef.current);
            }
        };
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            // Check file size
            if (file.size > MAX_FILE_SIZE) {
                alert("File is too large. Please upload a smaller JSON file.");
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const json = JSON.parse(e.target.result);
                    setJsonData(json);
                    event.target.value = null;
                } catch (error) {
                    console.error("Error parsing JSON:", error);
                    alert("Invalid JSON file.");
                }
            };
            setFileNameUpload(file.name);
            reader.readAsText(file);
            // console.log(jsonData);
        }
    };

    const [selections, setSelections] = useState({
        dropdownTrnOrNot: "inf",
        dropdownFullOrNot: "full_trn",
        dropdownOpt: "adam_opt",
        dropdownQuant: "no_quant",
        dropdownGPU: "rtx-2060",
        dropdownCPU: "3600x",
        dropdownDDR: "ddr4",
        isGPUorCPU: "usingGPU",
    });

    function setDDROptions(value) {
        let cpuSpecs = cpuJSONData[value];
        // console.log("calling: ", cpuSpecs);
        if (cpuSpecs["ddr4"] == 1 && cpuSpecs["ddr5"] == 1) {
            setShowDDR([1, 1]);
            return;
        }
        if (cpuSpecs["ddr4"] == 1) {
            setShowDDR([1, 0]);
            return;
        }
        if (cpuSpecs["ddr5"] == 1) {
            setShowDDR([0, 1]);
            return;
        }
    }

    const handleChangeSelection = (e) => {
        const { name, value } = e.target;
        setSelections((prevState) => ({
            ...prevState,
            [name]: value,
        }));

        if (name === "dropdownCPU") {
            setDDROptions(value);
        }
    };

    // const handleChangeInText1 = (event) => {
    //   setModelSize(event.target.value);
    // };

    const [output1, setOutput1] = useState("");

    function enchanceGPUJSONData(onlyNumGPUJsonData) {
        const newJsonData = {
            Name: selections.dropdownGPU.toUpperCase(),
            bandwidth: onlyNumGPUJsonData["bandwidth"] + " GB",
            compute: onlyNumGPUJsonData["compute"] + " TFlops/s",
            memory: onlyNumGPUJsonData["memory"] + " GB",
        };
        return newJsonData;
    }

    function enchanceCPUJSONData(onlyNumCPUJsonData) {
        const newJsonData = {
            Name: selections.dropdownCPU.toUpperCase(),
            "DDR5 Rated Speed": onlyNumCPUJsonData["Speed"] + " MT/s",
            "DDR4 Rated Speed": onlyNumCPUJsonData["speed_ddr4"] + " MT/s",
            Cores: onlyNumCPUJsonData["Cores"],
            "DDR5 Support": Boolean(onlyNumCPUJsonData["ddr5"]).toString(),
            "DDR4 Support": Boolean(onlyNumCPUJsonData["ddr4"]).toString(),
            "Memory Bus": onlyNumCPUJsonData["Bus"] + " Channel",
        };
        // console.log("My data");
        // console.log(newJsonData);
        return newJsonData;
    }

    // function getTotalFlops(parsedConfig){

    //     let totalFlops = 0;
    //     totalFlops += vocab*hiddenDim*2; //embedding
    //     totalFlops += hiddenDim*hiddenDim*2 //qkvo

    // }

    function getTotalFlopsForKV(parsedConfig, batchSize, contextLen) {
        const hidDim = parsedConfig["hiddenDim"];
        return 2 * contextLen * contextLen * hidDim * batchSize;
    }

    function convertGBToByte(sizeInGB) {
        return sizeInGB * 1024 * 1024 * 1024;
    }

    function convertByteToGB(sizeInByte) {
        return sizeInByte / (1024 * 1024 * 1024);
    }

    function convertByteToMB(sizeInByte) {
        return sizeInByte / (1024 * 1024);
    }

    function getFloatRatio_F16(quant) {
        return 1.0;
    }

    function getCPUSpeedFromSpecs(speed, speed_ddr4, bus, memory) {
        const busMap = { Dual: 2.0, Quad: 4.0, Hexa: 6.0, Octa: 8.0 };

        // console.log("speeds: ",speed, speed_ddr4, selections.dropdownDDR);

        let useThiSpeed = 0;
        if (selections.dropdownDDR==='ddr4'){
            useThiSpeed = speed_ddr4;
        }
        else{
            useThiSpeed = speed;
        }

        const busValue = busMap[bus];
        const rateMult = 8.0;

        const memInGBPerSecond = (busValue * rateMult * useThiSpeed) / 1024;

        return memInGBPerSecond;
    }

    function getFloatRatio_F16_CPU(quantType) {
        let k_values = [2, 3, 4, 5, 6, 8, 16];
        for (let k of k_values) {
            if (quantType.includes(k.toString())) {
                return k / 16;
            }
        }
        return 1.0;
    }

    function token_per_second_logic_CPU(
        cpuDataOnlyNum,
        parsedJSONData,
        promptLen,
        contextLen,
        batchSize,
        setErrorMessage,
        openModal
    ) {
        
        const speed = cpuDataOnlyNum["Speed"];
        const speed_ddr4 = cpuDataOnlyNum["speed_ddr4"];

        const bus = cpuDataOnlyNum["Bus"];
        const memory = cpuDataOnlyNum["Memory"];
        const cpu_compute = cpuDataOnlyNum["Flops"] * 0.5;

        const cpu_bandwidth = getCPUSpeedFromSpecs(speed, speed_ddr4, bus, memory);

        const quantType = selections.dropdownQuant;

        let parsedConfig = getParseConfig(
            parsedJSONData,
            setErrorMessage,
            openModal
        );
        const numLayers = parsedConfig["num_layers"],
            hiddenDim = parsedConfig["hiddenDim"];

        let memoryTransfer =
            (computeModelSizeGGML(parsedConfig, quantType) * 1024 * 1024) / 2.0;
        if (quantType === "no_quant") {
            memoryTransfer = computeModelSize(parsedConfig);
        }

        

        const extraFactorCPU = 1.6;
        //! Prompt Time Calculation
        //Time to process prompt (depending on contextLen this is either compute bound or memory bound)
        //Since the prompts are usually (above >50, i think it is safe to say this is mostly COMPUTE BOUND)

        // console.log("this is memory: ",convertByteToMB(memoryTransfer),quantType);
        const totalLen = parseInt(contextLen) + parseInt(promptLen);
        // console.log(
        //     "Theory: ",
        //     promptLen,
        //     memoryTransfer,
        //     numLayers,
        //     hiddenDim,
        //     batchSize
        // );
        let theoryTimePrompt =
            2 * promptLen * memoryTransfer +
            2 * numLayers * hiddenDim * hiddenDim * 2 * promptLen;
        theoryTimePrompt = batchSize * theoryTimePrompt;

        // console.log("first: ", theoryTimePrompt);
        let theoryTimePrompt_in_ms =
            theoryTimePrompt / (tera * (cpu_compute / 1000.0));

        // console.log("first: ",theoryTimePrompt_in_ms)
        console.log("mem trans: ", convertByteToMB(memoryTransfer));
        let finalPromptTime =
            theoryTimePrompt_in_ms * getFloatRatio_F16_CPU(quantType) +
            convertByteToMB(2 * memoryTransfer) * (0.008 / 1000);

        // const totalFlopsInB = 2*batchSize*modelSizeinB*billion + getTotalFlopsForKV(parsedConfig, batchSize, contextLen);

        //! Per token Time calculation
        const utilizationRate = 1.0;
        const kv_cache_memory = 2 * 2 * numLayers * hiddenDim * totalLen;

        //! Why is this 2* factor here? because of float16? -> Yes!
        let timeIfMemory =
            (convertByteToGB(2 * memoryTransfer + kv_cache_memory) /
                (utilizationRate * cpu_bandwidth)) *
            extraFactorCPU;
        let timeIfMemory_in_ms = timeIfMemory * 1000;

        //! Check if it is compute bound

        // console.log(
        //     memoryTransfer,
        //     numLayers,
        //     hiddenDim,
        //     batchSize,
        //     cpu_compute,
        //     extraFactorCPU
        // );
        let totalFlopsToken =
            2 * memoryTransfer +
            2 * totalLen * hiddenDim * 2 * numLayers * 2 * 2;
        totalFlopsToken = batchSize * totalFlopsToken;
        let timeIfFlops_in_ms =
            (totalFlopsToken * 1000) / (tera * (cpu_compute / 1000.0));
        timeIfFlops_in_ms = timeIfFlops_in_ms * extraFactorCPU;

        let finalTimeToConsider = null;
        let memoryOrCompute = null;

        if (timeIfMemory_in_ms > timeIfFlops_in_ms) {
            finalTimeToConsider = timeIfMemory_in_ms;
            memoryOrCompute = "memory";
        } else {
            finalTimeToConsider = timeIfFlops_in_ms;
            memoryOrCompute = "compute";
        }

        let token_per_s = 1000 / finalTimeToConsider; //finalTimeToConsider is time in ms for each token. So divide by 1000

        setComputedTokenPerSecond(Math.round(token_per_s));

        const jsonComputeReturnData = {
            "Token/s":
                Math.round(token_per_s) >= 1 ? Math.round(token_per_s) : "< 1",
            "ms per token": finalTimeToConsider.toFixed(2),
            // "ms per token (compute bound)": timeIfFlops_in_ms.toFixed(2),
            "Prompt process Time (s)": finalPromptTime.toFixed(2),
            "memory or compute bound?": memoryOrCompute,
        };

        setJsonDataCompute(jsonComputeReturnData);
        setShowTableCompute(true);
    }

    function token_per_second_logic_Train(
        gpuDataOnlyNum,
        parsedJSONData,
        promptLen,
        contextLen,
        batchSize,
        setErrorMessage,
        openModal
    ) {
        //! Training is most of the time compute bound
        const gpu_bandwidth = gpuDataOnlyNum["bandwidth"];
        const gpu_compute = gpuDataOnlyNum["compute"];

        const trnType = selections.dropdownTrnOrNot;
        const quantType = selections.dropdownQuant;
        const totalLen = parseInt(promptLen) + parseInt(contextLen);

        setShowTableComputeDisclaimer("");
        let bnb_cost = 1.0;
        if (quantType === "bnb_int8") {
            setShowTableComputeDisclaimer(
                "Disclaimer: bitsandbytes llm.int8 quant is NOT optimized for time. It takes more time than float16"
            );
            bnb_cost = 3.0;
        }
        if (quantType === "bnb_q4") {
            setShowTableComputeDisclaimer(
                "Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4/qlora is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues. "
            );
            bnb_cost = 2.75;
        }
        if (quantType === "qlora") {
            setShowTableComputeDisclaimer(
                "Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4/qlora is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues. "
            );
            bnb_cost = 1.75;
        }

        let parsedConfig = getParseConfig(
            parsedJSONData,
            setErrorMessage,
            openModal
        );
        const numLayers = parsedConfig["num_layers"],
            hiddenDim = parsedConfig["hiddenDim"];

        const memoryTransfer = computeModelSize(parsedConfig);

        let totalFlopsToken =
            2 * batchSize * totalLen * memoryTransfer +
            totalLen * hiddenDim * 2 * numLayers * batchSize;

        // console.log(batchSize, totalLen, memoryTransfer);
        // console.log(
        //     "other: ",
        //     totalLen * hiddenDim * 2 * numLayers * batchSize
        // );

        // console.log(
        //     2 * memoryTransfer,
        //     totalLen * hiddenDim * 2 * numLayers * 2
        // );

        let extraGradChoice = 1.0;
        if (selections.dropdownOpt === "adam_opt") {
            extraGradChoice = 1.15;
        }

        console.log("tot flops: ", totalFlopsToken);

        totalFlopsToken = totalFlopsToken * 2; //! Backward pass *2
        totalFlopsToken = totalFlopsToken * extraGradChoice;

        totalFlopsToken = totalFlopsToken * bnb_cost; //! Cost due to bnb

        if (selections.dropdownFullOrNot === "full_trn") {
            //! In total training, we will have to move the weights back to GPU for update, so its 2x more + update all so 1.5x (approx) more. Total 3x
            totalFlopsToken = totalFlopsToken * 3; //! I don't have capcacity to check this
        }

        let timeIfFlops_in_ms =
            (totalFlopsToken * 1000) / (tera * gpu_compute * 0.85);
        let memoryOrCompute = "compute";
        const jsonComputeReturnData = {
            "ms per iteration(forward + backward)":
                timeIfFlops_in_ms.toFixed(2),
            "memory or compute bound?": memoryOrCompute,
        };

        // console.log(jsonComputeReturnData);

        setJsonDataCompute(jsonComputeReturnData);
        setShowTableCompute(true);
    }

    function token_per_second_logic_GPU(
        gpuDataOnlyNum,
        parsedJSONData,
        promptLen,
        contextLen,
        batchSize,
        setErrorMessage,
        openModal
    ) {
        const gpu_bandwidth = gpuDataOnlyNum["bandwidth"];
        const gpu_compute = gpuDataOnlyNum["compute"];

        const trnType = selections.dropdownTrnOrNot;
        const quantType = selections.dropdownQuant;
        const totalLen = parseInt(promptLen) + parseInt(contextLen);

        let extraFactor = 1.0;

        if (trnType === "inf") {
            extraFactor = 2.0;
        }
        if (trnType === "inf_ggml") {
            extraFactor = 1.5;
            if (quantType === "ggml_Q2_K") {
                extraFactor = 2.0;
            }
        }

        if ((trnType === "inf") & (selections.dropdownFullOrNot === "qlora")) {
            setErrorMessage(
                "afaik qlora trained model's inference is just 4 bit inference, i.e. bnb int4/nf4. You can select that option from quant to calculate this"
            );
            openModal();
            return;
        }

        setShowTableComputeDisclaimer("");
        let bnb_cost = 1.0;
        if (trnType === "inf" && quantType === "bnb_int8") {
            setShowTableComputeDisclaimer(
                "Disclaimer: bitsandbytes llm.int8 quant is NOT optimized for inference. It takes more than time than float16."
            );
            bnb_cost = 4.5;
        }
        if (trnType === "inf" && quantType === "bnb_q4") {
            setShowTableComputeDisclaimer(
                "Disclaimer: https://github.com/TimDettmers/bitsandbytes/releases/tag/0.41.0 says that int4 is 2-4x faster but I haven't been able to reproduce this. Other people have raised similar issues in the repo."
            );
            bnb_cost = 3.0;
        }

        let parsedConfig = getParseConfig(
            parsedJSONData,
            setErrorMessage,
            openModal
        );
        const numLayers = parsedConfig["num_layers"],
            hiddenDim = parsedConfig["hiddenDim"];

        let memoryTransfer = 0;
        if (ggml_quants.includes(quantType)) {
            memoryTransfer =
                (computeModelSizeGGML(parsedConfig, quantType) * 1024 * 1024) /
                2.0;
        } else {
            if (quantType === "no_quant") {
                memoryTransfer = computeModelSize(parsedConfig);
            } else {
                if (quantType === "bnb_int8") {
                    memoryTransfer = computeModelSize(parsedConfig) / 2.0;
                }
                if (quantType === "bnb_q4") {
                    memoryTransfer = computeModelSize(parsedConfig) / 4.0;
                }
            }
        }

        //! Prompt Time Calculation
        //Time to process prompt (depending on contextLen this is either compute bound or memory bound)
        //Since the prompts are usually (above >50, i think it is safe to say this is mostly COMPUTE BOUND)

        let theoryTimePrompt =
            2 * promptLen * memoryTransfer +
            2 * numLayers * hiddenDim * hiddenDim * 2 * promptLen;
        theoryTimePrompt = batchSize * theoryTimePrompt;
        let theoryTimePrompt_in_ms =
            theoryTimePrompt / (tera * gpu_compute * 0.85);

        let finalPromptTime =
            theoryTimePrompt_in_ms * getFloatRatio_F16(quantType) * 1.8 +
            convertByteToMB(2 * memoryTransfer) * (0.008 / 100);

        // const totalFlopsInB = 2*batchSize*modelSizeinB*billion + getTotalFlopsForKV(parsedConfig, batchSize, contextLen);

        //! Per token Time calculation
        const utilizationRate = 1.0;
        const kv_cache_memory = 2 * 2 * numLayers * hiddenDim * totalLen;

        // console.log(
        //     "memory GPU side: ",
        //     convertByteToMB(memoryTransfer),
        //     memoryTransfer
        // );

        //1326940160*2

        let timeIfMemory =
            convertByteToGB(
                2 * memoryTransfer * extraFactor + kv_cache_memory * extraFactor
            ) /
            (utilizationRate * gpu_bandwidth);
        let timeIfMemory_in_ms = timeIfMemory * 1000;

        //! Check if it is compute bound
        let totalFlopsToken =
            2 * memoryTransfer + totalLen * hiddenDim * 2 * numLayers * 2 * 2;

        // console.log(
        //     2 * memoryTransfer,
        //     totalLen * hiddenDim * 2 * numLayers * 2
        // );

        totalFlopsToken = batchSize * totalFlopsToken;
        let timeIfFlops_in_ms =
            (totalFlopsToken * 1000) / (tera * gpu_compute * 0.85);

        let finalTimeToConsider = null;
        let memoryOrCompute = null;

        if (timeIfMemory_in_ms > timeIfFlops_in_ms) {
            finalTimeToConsider = timeIfMemory_in_ms;
            memoryOrCompute = "memory";
        } else {
            finalTimeToConsider = timeIfFlops_in_ms;
            memoryOrCompute = "compute";
        }

        if (!isValidPositiveInteger(numGPU)) {
            setErrorMessage("Number of GPUs have to be positive number (>0)");
            openModal();
            return;
        }

        if (numGPU > 1) {
            finalTimeToConsider = (finalTimeToConsider * 1.25) / numGPU;
        }

        finalTimeToConsider = finalTimeToConsider * bnb_cost;
        finalPromptTime = finalPromptTime * bnb_cost;

        let token_per_s = 1000 / finalTimeToConsider; //finalTimeToConsider is time in ms for each token. So divide by 1000

        setComputedTokenPerSecond(Math.round(token_per_s));

        const jsonComputeReturnData = {
            "Token/s":
                Math.round(token_per_s) >= 1 ? Math.round(token_per_s) : "< 1",
            "ms per token": finalTimeToConsider.toFixed(2),
            // "ms per token (compute bound)": timeIfFlops_in_ms.toFixed(2),
            "Prompt process Time (s)": finalPromptTime.toFixed(2),
            "memory or compute bound?": memoryOrCompute,
        };

        setJsonDataCompute(jsonComputeReturnData);
        setShowTableCompute(true);
    }

    function showGPUSpecs() {
        const gpuDataOnlyNum = gpuJSONData[selections.dropdownGPU];
        setGPUJSONDataForTable(enchanceGPUJSONData(gpuDataOnlyNum));
        setShowTableGPU(true);
    }

    function showCPUSpecs() {
        const cpuDataOnlyNum = cpuJSONData[selections.dropdownCPU];
        setCPUJSONDataForTable(enchanceCPUJSONData(cpuDataOnlyNum));
        setShowTableCPU(true);
    }

    function sanityChecks() {
        

        if (!isValidPositiveInteger(batchSize)) {
            setErrorMessage(
                "Batch size cant have non numeric or negative/zero values"
            );
            openModal();
            return false;
        }

        let check1 = checkCombinationInferenceTok(
            selections.dropdownTrnOrNot,
            selections.dropdownQuant,
            setErrorMessage,
            openModal
        );


        let check2 = checkCombinationTrainInference(
            selections.dropdownQuant,
            setErrorMessage,
            openModal,
            selections.dropdownFullOrNot
        );

        return check1 && check2;
    }

    function handleClickTokS() {
        // setErrorMessage("To be added");
        // openModal();
        if (
            !isValidPositiveInteger(contextLen) ||
            !isValidPositiveInteger(promptLen)
        ) {
            setErrorMessage(
                "context len & promt len should be positive numbers (>0)"
            );
            openModal();
            return;
        }

        if (!sanityChecks()) {
            return;
        }

        if (
            selections.isGPUorCPU === "usingCPU" &&
            selections.dropdownTrnOrNot != "inf_ggml"
        ) {
            setErrorMessage(
                "Inference with CPU only makes applicable(sensible) for GGML"
            );
            openModal();
            return;
        }

        if (selections.dropdownTrnOrNot === "inf_vLLM") {
            setErrorMessage(
                "Still working on adding vLLM. For now, as a rule of thumb, vLLM is 2-3x faster (than HF) when serving requests at your GPUs capacity"
            );
            openModal();
            return;
        }

        // if (selections.dropdownTrnOrNot === "trn") {
        //     setErrorMessage(
        //         "Token/s doesn't make sense for training, as whole sequence is generated at once. But how much time will one forward/backward pass take makese sense. I haven't added that yet."
        //     );
        //     openModal();
        //     return;
        // }
        if (
            selections.dropdownTrnOrNot === "trn" &&
            selections.isGPUorCPU === "usingCPU"
        ) {
            setErrorMessage("You can't train using HuggingFace on CPU");
            openModal();
            return;
        }

        // console.log(gpuJSONData);
        // console.log(cpuJSONData);
        // console.log(selections.dropdownGPU);

        const gpuDataOnlyNum = gpuJSONData[selections.dropdownGPU];
        const cpuDataOnlyNum = cpuJSONData[selections.dropdownCPU];

        let parsedConfig = responseCache.hasOwnProperty(modelName)
            ? responseCache[modelName]
            : null;

        if (parsedConfig === null) {
            setErrorMessage("Huggingface ID not present");
            openModal();
            return;
        }


        if (selections.dropdownTrnOrNot === "trn") {
            token_per_second_logic_Train(
                gpuDataOnlyNum,
                parsedConfig,
                promptLen,
                contextLen,
                batchSize,
                setErrorMessage,
                openModal
            );
            setCompReqHardwareName(selections.dropdownGPU);
            setShowTableComputeSmallInfo(2);
            setCompReqHardwareNameBefore("Time for training: ");
            return;
        }

        if (selections.isGPUorCPU === "usingGPU") {
            //! If I have bnb4 or bnb8 selected then put a disclaimer that it doesn't work

            token_per_second_logic_GPU(
                gpuDataOnlyNum,
                parsedConfig,
                promptLen,
                contextLen,
                batchSize,
                setErrorMessage,
                openModal
            );
            setCompReqHardwareName(selections.dropdownGPU);
            setShowTableComputeSmallInfo(1);
            setCompReqHardwareNameBefore("Tokens/s stats for: ");
        } else {
            token_per_second_logic_CPU(
                cpuDataOnlyNum,
                parsedConfig,
                promptLen,
                contextLen,
                batchSize,
                setErrorMessage,
                openModal
            );
            setCompReqHardwareName(selections.dropdownCPU);
            setShowTableComputeSmallInfo(1);
            setCompReqHardwareNameBefore("Tokens/s stats for: ");
        }
        return;
    }

    async function handleReset() {
        setFileNameUpload("");
        setJsonData(null);
        // setTotalMemoryShown("");
        // setBreakDownMemory("");
        setContextLen(1);
        setPromptLen(1);
        setShowTableGPU(false);
        setShowTable(false);
        setBatchSize("");
        setModelSize("");
        setModelName("");
        setShowTableCPU(false);
        setShowTableCompute(false);
    }

    async function handleClick() {
        if (modelName.includes("GGML") || modelName.includes("GGUF")) {
            setErrorMessage(
                "If you want info about GGML/GGUF models then enter the normal name & select GGML inference & quant type below. For example, if you want info about llama-2-7b.Q3_K_L.gguf then enter meta-llama/Llama-2-7b in the model name"
            );
            openModal();
            return;
        }
        let parsedConfig = responseCache.hasOwnProperty(modelName)
            ? responseCache[modelName]
            : null;

        if (
            !isValidPositiveInteger(contextLen) ||
            !isValidPositiveInteger(promptLen)
        ) {
            setErrorMessage(
                "context len & promt len should be positive numbers (>0)"
            );
            openModal();
        }

        const out = getAllComputedData(
            parsedConfig,
            jsonData,
            modelSize,
            parseInt(contextLen) + parseInt(promptLen),
            2,
            selections,
            setErrorMessage,
            openModal,
            batchSize
        );

        if (out == null) {
            return;
        }

        // setTotalMemoryShown(`Total Memory: ${out["Total"]} MB`);
        // const jsonOut = JSON.stringify(out);
        // setBreakDownMemory(`Breakdown(in MB): ${jsonOut}`);
        setTotalMemoryShown(out["Total"]);

        setShowTable(true);

        // setGPUJSONDataForTable(
        //     enchanceGPUJSONData(gpuJSONData[selections.dropdownGPU])
        // );

        let numGPUsINeed = Math.ceil(
            out["Total"] /
                (1024 * gpuJSONData[selections.dropdownGPU]["memory"])
        );
        // const nameOfGPUForNeed = selections.dropdownGPU + ' GPUs Needed'
        setNumGPUINeed(numGPUsINeed);
        setMemReqHardwareName(selections.dropdownGPU);
        setBreakDownMemoryJson(out);
    }

    // const handleClick = () => {

    //   const trnVal = selections.dropdownTrnOrNot;
    //   let totalMemory = 0;
    //   let size = parseFloat(modelSize);
    //   if (trnVal==='trn'){

    //   }

    //   console.log(modelSize);
    //   console.log(isNumberOrFloat(modelSize));

    //   // console.log("clicking");
    //   // setOutput1(selections.dropdownTrnOrNot + ' ' + selections.dropdownFullOrNot);

    //   // console.log()

    // };

    useEffect(() => {
        // Your function here to populate myVariable
        const fetchData = async () => {

            // Fetch data or perform some other operation
            let response = await fetch(configPath);
            response = await response.json();
            setResponseCache(response);
            setResponseCacheKeys(Object.keys(response));
        };

        fetchData();
    }, []);

    useEffect(() => {
        if (modelName && responseCacheKeys) {
            if (modelName.length > 1) {
                const filtered = responseCacheKeys.filter((item) =>
                    item.startsWith(modelName)
                );
                setSuggestions(filtered.slice(0, 10));
            } else {
                setSuggestions([]);
            }
        } else {
            setSuggestions([]);
        }
    }, [modelName]);

    // useEffect(() => {
    //     if (modelName) {
    //         if (modelName.length > 2) {
    //             const filtered = responseCacheKeys.filter((item) =>
    //                 item.startsWith(modelName)
    //             );
    //             setSuggestions(filtered.slice(0, 10));
    //         } else {
    //             setSuggestions([]);
    //         }
    //     } else {
    //         setSuggestions([]);
    //     }
    // }, [modelName]);

    // console.log(responseCache);

    const handleKeyDown = (e) => {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setSelectedIdx((prevIdx) =>
                Math.min(prevIdx + 1, suggestions.length - 1)
            );
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setSelectedIdx((prevIdx) => Math.max(prevIdx - 1, -1));
        } else if (e.key === "Enter" && selectedIdx >= 0) {
            setModelName(suggestions[selectedIdx]);
            setShowSuggestions(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <div className="App">
                    <Modal
                        isOpen={modalIsOpen}
                        // onAfterOpen={afterOpenModal}
                        className="m-auto mt-24 w-3/4 md:w-1/2 lg:w-1/3 bg-white p-4 rounded shadow-xl"
                        overlayClassName="fixed inset-0 bg-black bg-opacity-50"
                        onRequestClose={closeModal}
                        contentLabel="Example Modal"
                    >
                        <div className="text-center justify-center content-center">
                            <button
                                className="border border-red-500 px-4 py-2 bg-red-100 text-red-700 hover:bg-red-200"
                                onClick={closeModal}
                            >
                                Close
                            </button>
                            <div className="text-bold">{errorMessage}</div>
                        </div>
                    </Modal>
                    <div className="pt-3 font-bold text-center font-poppins">
                        <span className="text-2xl">Are you GPU poor?</span>{" "}
                        <span className="text-2xl hover:text-3xl">🫵🤨</span>
                    </div>
                    <div className="text-center text-l font-poppins pb-1">
                        Calculate GPU memory and token/s for any LLM
                    </div>
                    <div className="flex pb-1 content-center justify-center">
                        <img
                            className="transform transition-transform duration-300 hover:scale-110 border border-gray-600 hover:border-2"
                            src="/gpu_poor/itsovermeme.png"
                            alt="meme"
                            style={{ width: "75px", height: "75px" }}
                        />
                        <p className="font-poppins pr-2 pl-2 pt-8">OR</p>
                        <img
                            className="transform transition-transform duration-300 hover:scale-110 border border-gray-600 hover:border-2"
                            src="/gpu_poor/weback.jpg"
                            alt="meme"
                            style={{ width: "75px", height: "75px" }}
                        />
                    </div>
                    <hr className="bg-gray-300"></hr>
                    <div className="flex flex-row mt-1">
                        <div>
                            <div className="border border-gray-400 p-3 rounded-lg inline-block hover:border-black">
                                <label className="text-sm font-poppins pr-4">
                                    Name (Hugginface ID)
                                </label>

                                <TextInput
                                    className="w-64 font-poppins input border border-black text-sm"
                                    value={modelName}
                                    setValue={setModelName}
                                    setChange={setShowSuggestions}
                                    handleKeyDown={handleKeyDown}
                                    placeholder="e.g. meta-llama/Llama-2-7b-hf"
                                />
                                {modelName && showSuggestions && (
                                    <ul className="mt-2 border rounded divide-y">
                                        {suggestions.map((item, index) => (
                                            <li
                                                key={index}
                                                onClick={() => {
                                                    setModelName(item);
                                                    setShowSuggestions(false);
                                                    
                                                }}
                                                className={`p-2 ${
                                                    selectedIdx === index
                                                        ? "bg-gray-300"
                                                        : "hover:bg-gray-200"
                                                } cursor-pointer`}
                                            >
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                )}

                                <div>
                                    <label className="text-sm">OR</label>
                                </div>
                                <div>
                                    <label className="pr-4 text-sm font-poppins">
                                        Size (in Billion)
                                    </label>
                                    <TextInput
                                        className="w-48 input border text-sm font-poppins border-black"
                                        value={modelSize}
                                        setValue={setModelSize}
                                        placeholder="e.g. for llama-7b enter 7"
                                    />
                                </div>
                                {/* <div className="text-sm pr-4 pb-1">OR</div> */}
                                {/* <div className="flex">
                            <div>
                                <input
                                    type="file"
                                    id="fileInput"
                                    accept=".json"
                                    onChange={handleFileChange}
                                    className="hidden"
                                />
                                <label
                                    htmlFor="fileInput"
                                    className="text-sm font-poppins px-1 py-1 bg-gray-200 border border-gray-300 cursor-pointer hover:bg-gray-300"
                                >
                                    Upload model config
                                </label>
                                <span className="text-sm font-serif underline">
                                    {fileNameUpload}
                                </span>
                            </div>
                            <div className="pl-6">
                                <button
                                    className="text-xs font-poppins bg-gray-100   border border-gray-300 cursor-pointer hover:bg-gray-300"
                                    onClick={handleFileClear}
                                >
                                    Clear file
                                </button>
                            </div>
                        </div> */}
                            </div>

                            <br></br>
                            <div className="border border-gray-400 p-2 rounded-lg inline-block hover:border-black mt-2">
                                <div className="pb-2">
                                    <label className="font-poppins text-sm pr-4">
                                        Train or Inference?
                                    </label>
                                    <select
                                        className="font-poppins text-sm border border-gray-500"
                                        name="dropdownTrnOrNot"
                                        onChange={handleChangeSelection}
                                    >
                                        <option value="inf">
                                            Inference (Huggingface)
                                        </option>
                                        <option value="inf_vLLM">
                                            Inference (vLLM)
                                        </option>
                                        {/* <option value="inf_exL">
                                        Inference (exLlama)
                                    </option> */}
                                        <option value="inf_ggml">
                                            Inference (GGML)
                                        </option>
                                        <option value="trn">
                                            Training (Huggingface)
                                        </option>
                                    </select>
                                </div>

                                <div className="flex pb-2 pt-1">
                                    <div className="pr-6">
                                        <label className="font-poppins text-sm pr-2">
                                            Train method?
                                        </label>
                                        <select
                                            className="font-poppins text-sm border border-gray-500"
                                            name="dropdownFullOrNot"
                                            onChange={handleChangeSelection}
                                        >
                                            <option value="full_trn">
                                                Full
                                            </option>
                                            <option value="lora_trn">
                                                LoRA
                                            </option>
                                            <option value="qlora">QLoRA</option>
                                        </select>
                                    </div>
                                    <div className="pr-6">
                                        <label className="text-sm pr-2 font-poppins">
                                            Optimizer?
                                        </label>
                                        <select
                                            className="text-sm font-poppins border border-gray-500"
                                            name="dropdownOpt"
                                            onChange={handleChangeSelection}
                                        >
                                            <option value="adam_opt">
                                                ADAM
                                            </option>
                                            <option value="sgd_opt">SGD</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="font-poppins text-sm pr-2">
                                            Quant?
                                        </label>
                                        <select
                                            className="font-poppins text-sm border border-gray-500"
                                            name="dropdownQuant"
                                            onChange={handleChangeSelection}
                                        >
                                            <option value="no_quant">
                                                None
                                            </option>
                                            <optgroup label="-----"></optgroup>
                                            <option value="bnb_int8">
                                                bnb int8
                                            </option>
                                            <option value="bnb_q4">
                                                bnb int4
                                            </option>

                                            <optgroup label="-----"></optgroup>
                                            <option value="ggml_Q2_K">
                                                GGML Q2_K
                                            </option>

                                            <option value="ggml_Q3_K_L">
                                                GGML Q3_K_L
                                            </option>
                                            <option value="ggml_Q3_K_M">
                                                GGML Q3_K_M
                                            </option>

                                            <option value="ggml_QK4_0">
                                                GGML QK4_0
                                            </option>
                                            <option value="ggml_QK4_1">
                                                GGML QK4_1
                                            </option>
                                            <option value="ggml_QK4_K_M">
                                                GGML QK4_K_M
                                            </option>
                                            <option value="ggml_QK4_K_S">
                                                GGML QK4_K_S
                                            </option>

                                            <option value="ggml_QK5_0">
                                                GGML QK5_0
                                            </option>
                                            <option value="ggml_QK5_1">
                                                GGML QK5_1
                                            </option>
                                            <option value="ggml_QK5_K_M">
                                                GGML QK5_K_M
                                            </option>

                                            <option value="ggml_Q6_K">
                                                GGML Q6_K
                                            </option>

                                            <option value="ggml_QK8_0">
                                                GGML QK8_0
                                            </option>
                                        </select>
                                    </div>
                                </div>

                                <div className="flex pt-1">
                                    <div>
                                        <label className="font-poppins text-sm">
                                            Prompt len?{" "}
                                        </label>
                                        <TextInput
                                            className="w-10 input border text-sm font-poppins border-black"
                                            setValue={setPromptLen}
                                            value={promptLen}
                                            placeholder="?"
                                        />
                                    </div>
                                    <div className="pl-3">
                                        <label className="font-poppins text-sm">
                                            Tokens to Generate?{" "}
                                        </label>
                                        <TextInput
                                            className="w-10 input border text-sm font-poppins border-black"
                                            setValue={setContextLen}
                                            value={contextLen}
                                            placeholder="?"
                                        />
                                    </div>
                                    <div className="pl-4">
                                        <label className="font-poppins text-sm">
                                            Batch-size?{" "}
                                        </label>
                                        <TextInput
                                            className="w-8 input border text-sm font-poppins border-black"
                                            setValue={setBatchSize}
                                            value={batchSize}
                                            placeholder="1"
                                        />
                                    </div>
                                </div>
                            </div>
                            <div className="flex flex-col border border-gray-400 p-2 rounded-lg mt-2 hover:border-black">
                            <div>
                            <label className="font-poppins font-bold text-sm pr-2">
                                        For Inference ⏱️
                                    </label>
                            </div>
                                <div className="pt-1">
                                    <label className="font-poppins font-extrabold text-sm pr-2">
                                        GPU or CPU?
                                    </label>
                                    <select
                                        className="font-poppins text-sm border border-gray-500"
                                        name="isGPUorCPU"
                                        onChange={handleChangeSelection}
                                    >
                                        <option value="usingGPU">GPU</option>
                                        <option value="usingCPU">CPU</option>
                                    </select>
                                    <label className="font-sans text-sm pl-2">
                                        
                                    </label>
                                </div>
                                {selections.isGPUorCPU === "usingGPU" && (
                                    <div className="flex pt-2 flex-row">
                                        <div>
                                            <label className="font-poppins font-semibold text-sm pr-2">
                                                GPU
                                            </label>
                                            <select
                                                className="font-poppins text-sm border border-gray-500"
                                                name="dropdownGPU"
                                                onChange={handleChangeSelection}
                                            >
                                                <option value="rtx-2060">
                                                    RTX 2060
                                                </option>
                                                <option value="rtx-2070">
                                                    RTX 2070
                                                </option>
                                                <option value="rtx-3060">
                                                    RTX 3060
                                                </option>
                                                <option value="rtx-3090">
                                                    RTX 3090
                                                </option>
                                                <option value="rtx-4060">
                                                    RTX 4060
                                                </option>
                                                <option value="rtx-4090">
                                                    RTX 4090
                                                </option>
                                                <option value="P-100 (12 GB)">
                                                    P 100
                                                </option>
                                                <option value="A-4000">
                                                    A 4000
                                                </option>
                                                <option value="A-6000">
                                                    A 6000
                                                </option>
                                            </select>
                                        </div>
                                        <div className="pl-5">
                                            <label className="font-poppins text-sm pr-2">
                                                No. of GPUs?
                                            </label>
                                            <TextInput
                                                className="w-8 input border text-sm font-poppins border-black"
                                                value={numGPU}
                                                setValue={setNumGPU}
                                                placeholder="1"
                                            />
                                        </div>
                                        <div>
                                            <button
                                                className="ml-5 border px-0.5 border-red-500 bg-red-100 font-poppins text-sm text-red-700 hover:bg-red-200"
                                                onClick={showGPUSpecs}
                                            >
                                                Get GPU specs
                                            </button>
                                        </div>
                                    </div>
                                )}

                                {selections.isGPUorCPU === "usingCPU" && (
                                    <div className="pt-2 flex flex-row">
                                        <div className="">
                                            <label className="font-poppins font-semibold text-sm pr-2">
                                                CPU
                                            </label>
                                            <select
                                                className="font-poppins text-sm border border-gray-500"
                                                name="dropdownCPU"
                                                onChange={handleChangeSelection}
                                            >
                                                <option value="3600x">
                                                    AMD 3600XT
                                                </option>
                                                <option value="7950x">
                                                    AMD 7950x
                                                </option>
                                                <option value="12700H">
                                                    i7-12700H
                                                </option>
                                                <option value="13900K">
                                                    i9-13900K
                                                </option>
                                                <option value="13700K">
                                                    i7-13700K
                                                </option>
                                                <option value="9900K">
                                                    i9 9900K
                                                </option>
                                                <option value="5900X">
                                                    AMD 5900X
                                                </option>
                                                <option value="5600X">
                                                    AMD 5600X
                                                </option>
                                                <option value="3990X">
                                                    AMD 3990X
                                                </option>
                                            </select>
                                        </div>
                                        {/* <div className="pl-5">
                                            <label className="font-poppins text-sm pr-2 hover:cursor-not-allowed">
                                                Layers to offload?{" "}
                                                <span className="text-red-800">
                                                    (to be added)
                                                </span>
                                            </label>
                                            <TextInput
                                                className="w-8 input border text-sm font-poppins border-black hover:cursor-not-allowed"
                                                value={numOffload}
                                                setValue={setNumOffLoad}
                                                placeholder="1"
                                                disableStatus={true}
                                            />
                                        </div> */}
                                        <div className="pl-5">
                                            <label className="font-poppins text-sm pr-2">
                                                RAM?{" "}
                                            </label>
                                            <select
                                                className="font-poppins text-sm border border-gray-500"
                                                name="dropdownDDR"
                                                onChange={handleChangeSelection}
                                            >
                                                {showDDR[0] === 1 && (
                                                    <option value="ddr4">
                                                        DDR4
                                                    </option>
                                                )}
                                                {showDDR[1] === 1 && (
                                                    <option value="ddr5">
                                                        DDR5
                                                    </option>
                                                )}
                                            </select>
                                        </div>
                                        <div>
                                            <button
                                                className="ml-5 border px-0.5 border-red-500 bg-red-100 font-poppins text-sm text-red-700 hover:bg-red-200"
                                                onClick={showCPUSpecs}
                                            >
                                                Get CPU specs
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div>
                                <br></br>
                                {/* <button className='bg-green-50' onClick={handleClick}>Generate Outputs</button> */}
                                <div className="flex">
                                    <div className="pr-6">
                                        <button
                                            class="font-poppins border text-sm border-blue-500 px-4 py-2 bg-blue-100 text-blue-700 hover:bg-blue-300"
                                            onClick={handleClick}
                                        >
                                            Find Memory requirement
                                        </button>
                                    </div>
                                    <div>
                                        <button
                                            class="font-poppins border text-sm border-green-500 px-4 py-2 bg-green-100 text-green-700 hover:bg-green-300"
                                            onClick={handleClickTokS}
                                        >
                                            Find ~tokens/s
                                        </button>
                                    </div>
                                    <div>
                                        <button
                                            class="font-poppins border text-sm ml-6 border-red-500 px-1 py-2 bg-red-100 text-red-700 hover:bg-red-300"
                                            onClick={handleReset}
                                        >
                                            CLEAR
                                        </button>
                                    </div>
                                    {/* <div className="pl-4 pt-1">
                                <button
                                    class="font-poppins border text-xs bg-gray-100 border-gray-500  text-black hover:bg-gray-300"
                                    onClick={handleReset}
                                >
                                    Reset
                                </button>
                            </div> */}
                                </div>
                            </div>
                        </div>
                        <div class="pl-4 border-l-2 border-gray-400 flex-shrink-0 ml-8">
                            <div className="font-poppins text-lg font-semibold">
                                How does
                                <span className="font-bold text-blue-400">
                                    {" "}
                                    X{" "}
                                </span>
                                tokens/s look like?
                            </div>
                            <div>
                                <div>
                                    <label className="pr-2 text-lg font-poppins">
                                        Enter Token/s:
                                    </label>
                                    <TextInput
                                        className="w-12 input border text-base font-poppins border-black"
                                        value={tokenPerSecond}
                                        setValue={setTokenPerSecond}
                                        placeholder="50"
                                    />
                                    <button
                                        className="ml-4 border px-0.5 border-red-500 bg-red-100 font-poppins text-red-700 hover:bg-red-200"
                                        onClick={handleClickGenerateText}
                                    >
                                        Generate Text
                                    </button>
                                    <button
                                        className="ml-4 border px-0.5 border-red-500 bg-red-100 font-poppins text-red-700 hover:bg-red-200"
                                        onClick={handleClearGeneratedText}
                                    >
                                        Clear
                                    </button>
                                </div>
                            </div>
                            <div className="pt-1 whitespace-normal overflow-hidden max-w-2xl font-poppins">
                                {isVisible && <div>{displayedText}</div>}
                            </div>
                        </div>
                    </div>
                    <br></br>
                    <hr className="h-1 bg-gray-100"></hr>
                    {/* <div className="font-bold font-poppins">
                        {totalMemoryShown}
                    </div> */}
                    <div className="flex flex-row">
                        <div className="px-2 py-1">
                            {showTable && (
                                <>
                                    <div className="text-sm font-poppins font-bold">
                                        Memory Requirement for:{" "}
                                        {/* {memReqHardwareName} */}
                                    </div>
                                    <table className="min-w-1/2 bg-white border-collapse border-2 border-black font-poppins">
                                        <tbody>
                                            {/* Total row */}
                                            <tr className="bg-blue-100 text-sm">
                                                <td className="py-1 px-2 font-bold border-black border-r-0">
                                                    Total
                                                </td>
                                                <td className="py-1 px-2 border">
                                                    <span className="font-bold">
                                                        {" "}
                                                        {
                                                            totalMemoryShown
                                                        } MB{" "}
                                                    </span>
                                                </td>
                                            </tr>

                                            {/* Breakdown row */}
                                            <tr>
                                                <td
                                                    className="text-xs text-center border-2 border-black"
                                                    colSpan="2"
                                                >
                                                    Breakdown
                                                </td>
                                            </tr>

                                            {/* Name-Value pairs */}
                                            {Object.entries(
                                                breakDownMemoryJson
                                            ).map(([key, value], index) => {
                                                if (key === "Total") {
                                                    return null; // Skip this iteration and return nothing
                                                }

                                                return (
                                                    <tr
                                                        className={`${
                                                            index % 2 === 0
                                                                ? "bg-blue-100"
                                                                : "bg-blue-50"
                                                        }`}
                                                        key={index}
                                                    >
                                                        <td className="py-1 px-2 text-sm border">
                                                            {key}
                                                        </td>
                                                        <td className="py-1 px-2 text-sm border">
                                                            {value} MB
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                            <tr className="bg-gray-200">
                                                <td className="py-1 px-2 text-sm border">
                                                    selected GPUs needed
                                                </td>
                                                <td className="py-1 px-2 text-sm border">
                                                    {numGPUINeed}
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </>
                            )}
                            {/* <button
                        className="mt-4 px-4 py-2 bg-blue-500 text-white"
                        onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(jsonData, null, 2));
                        }}
                    >
                        Copy to Clipboard
                    </button> */}
                        </div>
                        <div className="py-1">
                            {showTableCompute && (
                                <div>
                                    <div className="text-sm font-poppins font-bold">
                                        {compReqHardwareNameBefore}
                                        {compReqHardwareName}
                                    </div>
                                    {/* {selections.isGPUorCPU==='usingGPU' ? selections.dropdownGPU : selections.dropdownCPU} */}
                                    <table className="min-w-1/2 bg-white border-collapse border-2 border-black font-poppins">
                                        <tbody>
                                            {/* Name-Value pairs */}
                                            {Object.entries(
                                                jsonDataCompute
                                            ).map(([key, value], index) => {
                                                if (key === "Total") {
                                                    return null; // Skip this iteration and return nothing
                                                }

                                                return (
                                                    <tr
                                                        className={`${
                                                            index % 2 === 0
                                                                ? "bg-violet-100"
                                                                : "bg-violet-50"
                                                        }`}
                                                        key={index}
                                                    >
                                                        <td className="py-1 px-2 text-sm border">
                                                            {key}
                                                        </td>
                                                        <td className="py-1 px-2 text-sm border">
                                                            {value}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                    <div className="text-xs whitespace-normal overflow-hidden max-w-xl font-poppins text-red-500">
                                        {showTableComputeDisclaimer}
                                    </div>
                                    {showTableComputeSmallInfo == 1 && (
                                        <div className="text-xs font-poppins text-blue-700">
                                            Check above to see how{" "}
                                            {computedTokenPerSecond} token/s
                                            looks like
                                        </div>
                                    )}
                                    {showTableComputeSmallInfo == 2 && (
                                        <div className="text-xs font-poppins text-blue-700">
                                            For train, generate length = 1.
                                            Since training is next token pred.
                                            e.g. if u train on 500 len sequence
                                            then put 500 in prompt len.
                                        </div>
                                    )}
                                </div>
                            )}
                            {/* <button
                        className="mt-4 px-4 py-2 bg-blue-500 text-white"
                        onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(jsonData, null, 2));
                        }}
                    >
                        Copy to Clipboard
                    </button> */}
                        </div>
                    </div>
                    <div className="pt-4 flex flex-row">
                        <div className="px-2 py-1">
                            {showTableGPU && (
                                <>
                                    <div className="text-sm font-poppins font-bold">
                                        GPU Info:
                                    </div>
                                    <table className="min-w-1/2 bg-white border-collapse border-2 border-black font-poppins">
                                        <tbody>
                                            {/* Total row */}
                                            {/* Name-Value pairs */}
                                            {Object.entries(
                                                gpuJsonDataForTable
                                            ).map(([key, value], index) => {
                                                return (
                                                    <tr
                                                        className={`${
                                                            index % 2 === 0
                                                                ? "bg-teal-100"
                                                                : "bg-teal-50"
                                                        }`}
                                                        key={index}
                                                    >
                                                        <td className="py-1 px-2 text-sm border">
                                                            {key}
                                                        </td>
                                                        <td className="py-1 px-2 text-sm border">
                                                            {value}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </>
                            )}
                            {/* <button
                        className="mt-4 px-4 py-2 bg-blue-500 text-white"
                        onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(jsonData, null, 2));
                        }}
                    >
                        Copy to Clipboard
                    </button> */}
                        </div>
                        <div className="px-2 py-1">
                            {showTableCPU && (
                                <>
                                    <div className="text-sm font-poppins font-bold">
                                        CPU Info:
                                    </div>
                                    <table className="min-w-1/2 bg-white border-collapse border-2 border-black font-poppins">
                                        <tbody>
                                            {/* Total row */}
                                            {/* Name-Value pairs */}
                                            {Object.entries(
                                                cpuJsonDataForTable
                                            ).map(([key, value], index) => {
                                                return (
                                                    <tr
                                                        className={`${
                                                            index % 2 === 0
                                                                ? "bg-teal-100"
                                                                : "bg-teal-50"
                                                        }`}
                                                        key={index}
                                                    >
                                                        <td className="py-1 px-2 text-sm border">
                                                            {key}
                                                        </td>
                                                        <td className="py-1 px-2 text-sm border">
                                                            {value}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </>
                            )}
                            {/* <button
                        className="mt-4 px-4 py-2 bg-blue-500 text-white"
                        onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(jsonData, null, 2));
                        }}
                    >
                        Copy to Clipboard
                    </button> */}
                        </div>
                    </div>
                    {/* <div className="font-bold font-poppins">{breakDownMemory}</div> */}
                </div>
                <div>
                    <div>
                        <a
                            className="text-xs underline font-mono text-blue-600 hover:font-bold"
                            href="https://github.com/RahulSChand/gpu_poor/"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Github repo (documentation)
                        </a>
                    </div>
                    <div
                        className="cursor-pointer text-black"
                        onClick={() => setFaqOpen(!faqOpen)}
                    >
                        <h2 className="font-mono text-sm text-bold underline hover:font-extrabold">
                            Read FAQ 🔽
                        </h2>
                    </div>

                    {faqOpen && (
                        <ul className="list-disc pl-5 text-xs font-mono">
                            <li>
                                These are APPORXIMATE values. They can vary by
                                +-15% depending on your CPU, GPU, cuda version,
                                llama.cpp version, model etc.
                            </li>
                            {/* <li>
                                For training, the total context length will be
                                prompt Len + Generate Len. The correct (ideal)
                                use case is to set generate = 1 for training,
                                since all training is next token prediction
                                loss.
                            </li> */}
                            <li>
                                CPU inference is only compatible with GGML. You
                                can't use CPU with HF/vLLM
                            </li>
                            <li>GPU + CPU is not yet supported</li>
                        </ul>
                    )}
                </div>

                {/* <div className="text-xs text-gray-600 font-semibold">
                    PS: These are approximate values & may vary by 500MB-1GB
                    depending on the GPU, model, input, cuda version etc. If
                    your setup has ~1GB over the requirement you should likely
                    be good.
                </div>
                <div>
                    <a
                        className="text-base underline text-blue-600 hover:font-bold"
                        href="https://github.com/RahulSChand/gpu_poor/"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        FAQ
                    </a>
                </div> */}
                {/* <button>Show Values</button>
      <input type="text" value={output1} readOnly />
      <input type="text" value={output2} readOnly /> */}
            </header>
        </div>
    );
}

export default App;

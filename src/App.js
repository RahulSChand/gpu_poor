import React, { useState } from "react";
import TextInput from "./textBox";
import Modal from "react-modal";

const billion = 1000000000;
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
        console.log("here!");
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
    const constant_4_extra = 1.5;
    const constant_qlora = 0.75;

    const common =
        (10 * parsedConfig.hiddenDim +
            5 * parsedConfig.hiddenDim +
            4 * parsedConfig.interDim +
            2 * parsedConfig.interDim) *
        parsedConfig.num_layers;

    let extra_mem = 0;

    if (quant === "bnb_int8") {
        extra_mem = constant_8_extra * common * contextLen;
    }

    if (quant === "bnb_q4") {
        extra_mem = constant_4_extra * common * contextLen;
        
    }

    if (quant === "qlora") {
        extra_mem = constant_qlora * common * contextLen;
        
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
    ){

    //! Can't train full with QLoRA
    if ((typeOfTrn==='full_trn') && ggml_quants.includes(quantType)){
        setErrorMessage("Can't use GGML for training");
        openModal();
        return false;
    }
    if (typeOfTrn==="qlora" && quantType!='no_quant'){
        setErrorMessage("QLoRA is 4bit explicit. No need to select a quant type if you are training using QLoRA. Set it to 'None'");
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
        (quantType === "bnb_int8" ||
            quantType === "bnb_q4")
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

    console.log(jsonUploadedData);

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

    let vocab = 32000;
    let heads = 32;
    let numLayers = 32;

    //vocab*h + numLayers*4*h*h + 3*4*h*h*numLayers = modelSize*10^9
    const A = numLayers * 4 + 3 * 4 * numLayers;
    const B = vocab;
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
    let modelSizeinMB = convertToMBModelSize(modelSizeinB, quantType, typeOfTrn);
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

        let checkSanity = checkCombinationTrainInference(quantType, setErrorMessage, openModal, typeOfTrn);
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
        console.log("got activation", activationMemory);

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
    return Number.isInteger(num) && num > 0 && input.trim() !== "";
}

function App() {
    // let subtitle;
    const [modelSize, setModelSize] = useState("");
    const [modelName, setModelName] = useState("");
    const [contextLen, setContextLen] = useState("");
    const [batchSize, setBatchSize] = useState("");
    const [totalMemoryShown, setTotalMemoryShown] = useState(" ");
    const [breakDownMemory, setBreakDownMemory] = useState(" ");
    const [errorMessage, setErrorMessage] = useState("");

    const [fileNameUpload, setFileNameUpload] = useState("");

    const [modalIsOpen, setIsOpen] = React.useState(false);

    const [jsonData, setJsonData] = useState(null);

    function openModal() {
        setIsOpen(true);
    }

    function closeModal() {
        setIsOpen(false);
    }

    const handleFileClear = (event) => {
        setFileNameUpload("");
        setJsonData(null);
        setTotalMemoryShown("");
        setBreakDownMemory("");
    };

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
            console.log(jsonData);
        }
    };

    const [selections, setSelections] = useState({
        dropdownTrnOrNot: "inf",
        dropdownFullOrNot: "full_trn",
        dropdownOpt: "adam_opt",
        dropdownQuant: "no_quant",
        dropdownGPU: "rtx_3090",
    });

    const handleChangeSelection = (e) => {
        const { name, value } = e.target;
        setSelections((prevState) => ({
            ...prevState,
            [name]: value,
        }));
    };

    // const handleChangeInText1 = (event) => {
    //   setModelSize(event.target.value);
    // };

    const [output1, setOutput1] = useState("");

    async function handleClickTokS() {
        setErrorMessage("To be added");
        openModal();
        return;
    }

    async function handleReset() {
        setFileNameUpload("");
        setJsonData(null);
        setTotalMemoryShown("");
        setBreakDownMemory("");
        setContextLen("");
        setBatchSize("");
        setModelSize("");
        setModelName("");
    }

    async function handleClick() {
        if (modelName.includes("GGML") || modelName.includes("GGUF")) {
            setErrorMessage(
                "If you want info about GGML/GGUF models then enter the normal name & select GGML inference & quant type below. For example, if you want info about llama-2-7b.Q3_K_L.gguf then enter meta-llama/Llama-2-7b in the model name"
            );
            openModal();
            return;
        }
        let parsedConfig = await fetchParams(specialMapping(modelName));
        const out = getAllComputedData(
            parsedConfig,
            jsonData,
            modelSize,
            contextLen,
            2,
            selections,
            setErrorMessage,
            openModal,
            batchSize
        );

        if (out == null) {
            return;
        }

        setTotalMemoryShown(`Total Memory: ${out["Total"]} MB`);
        const jsonOut = JSON.stringify(out);
        setBreakDownMemory(`Breakdown(in MB): ${jsonOut}`);
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
                    <div className="pt-3 font-bold text-center font-mono">
                        <span className="text-2xl">Are you GPU poor?</span>{" "}
                        <span className="text-2xl hover:text-3xl">🫵🤨</span>
                    </div>
                    <div className="text-center text-l font-mono pb-2">
                        Calculate how much GPU memory you need to run a
                        particular LLM
                    </div>
                    <div className="flex pb-1 content-center justify-center">
                        <img
                            className="transform transition-transform duration-300 hover:scale-110 border border-gray-600 hover:border-2"
                            src="/gpu_poor/itsovermeme.png"
                            alt="meme"
                            style={{ width: "125px", height: "125px" }}
                        />
                        <p className="font-mono pr-2 pl-2 pt-8">OR</p>
                        <img
                            className="transform transition-transform duration-300 hover:scale-110 border border-gray-600 hover:border-2"
                            src="/gpu_poor/weback.jpg"
                            alt="meme"
                            style={{ width: "125px", height: "125px" }}
                        />
                    </div>
                    <div className="border border-gray-400 p-4 rounded-lg inline-block hover:border-black">
                        <div>
                            <label className="text-sm font-mono pr-4">
                                Model Name (Hugginface ID)
                            </label>
                            <TextInput
                                className="w-64 font-mono input border border-black text-sm"
                                value={modelName}
                                setValue={setModelName}
                                placeholder="e.g. meta-llama/Llama-2-7b-hf"
                            />
                        </div>
                        <label className="text-sm">OR</label>

                        <div>
                            <label className="pr-4 text-sm font-mono">
                                Model Size (in Billion)
                            </label>
                            <TextInput
                                className="w-64 input border text-sm font-mono border-black"
                                value={modelSize}
                                setValue={setModelSize}
                                placeholder="e.g. for llama-7b enter 7"
                            />
                        </div>
                        <div className="text-sm pr-4 pb-1">OR</div>
                        <div className="flex">
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
                                    className="text-sm font-mono px-1 py-1 bg-gray-200 border border-gray-300 cursor-pointer hover:bg-gray-300"
                                >
                                    Upload model config
                                </label>
                                <span className="text-sm font-serif underline">
                                    {fileNameUpload}
                                </span>
                            </div>
                            <div className="pl-6">
                                <button
                                    className="text-xs font-mono bg-gray-100   border border-gray-300 cursor-pointer hover:bg-gray-300"
                                    onClick={handleFileClear}
                                >
                                    Clear file
                                </button>
                            </div>
                        </div>
                    </div>

                    <br></br>

                    <div className="pb-2 pt-1">
                        <label className="font-mono text-sm pr-4">
                            Training or Inference?
                        </label>
                        <select
                            className="font-mono text-sm border border-gray-500"
                            name="dropdownTrnOrNot"
                            onChange={handleChangeSelection}
                        >
                            <option value="inf">Inference (Huggingface)</option>
                            <option value="inf_vLLM">Inference (vLLM)</option>
                            <option value="inf_exL">Inference (exLlama)</option>
                            <option value="inf_ggml">Inference (GGML)</option>
                            <option value="trn">Training (Huggingface)</option>
                        </select>
                    </div>

                    <div className="flex pb-2">
                        <div className="pr-6">
                            <label className="font-mono text-sm pr-2">
                                Training method?
                            </label>
                            <select
                                className="font-mono text-sm border border-gray-500"
                                name="dropdownFullOrNot"
                                onChange={handleChangeSelection}
                            >
                                <option value="full_trn">Full</option>
                                <option value="lora_trn">LoRA</option>
                                <option value="qlora">QLoRA</option>
                            </select>
                        </div>
                        <div className="pr-6">
                            <label className="text-sm pr-2 font-mono">
                                Optimizer?
                            </label>
                            <select
                                className="text-sm font-mono border border-gray-500"
                                name="dropdownOpt"
                                onChange={handleChangeSelection}
                            >
                                <option value="adam_opt">ADAM</option>
                                <option value="sgd_opt">SGD</option>
                            </select>
                        </div>
                        <div>
                            <label className="font-mono text-sm pr-2">
                                Quantization?
                            </label>
                            <select
                                className="font-mono text-sm border border-gray-500"
                                name="dropdownQuant"
                                onChange={handleChangeSelection}
                            >
                                <option value="no_quant">None</option>
                                <optgroup label="-----"></optgroup>
                                <option value="bnb_int8">bnb int8</option>
                                <option value="bnb_q4">bnb int4</option>
                                
                                <optgroup label="-----"></optgroup>
                                <option value="ggml_Q2_K">GGML Q2_K</option>

                                <option value="ggml_Q3_K_L">GGML Q3_K_L</option>
                                <option value="ggml_Q3_K_M">GGML Q3_K_M</option>

                                <option value="ggml_QK4_0">GGML QK4_0</option>
                                <option value="ggml_QK4_1">GGML QK4_1</option>
                                <option value="ggml_QK4_K_M">
                                    GGML QK4_K_M
                                </option>
                                <option value="ggml_QK4_K_S">
                                    GGML QK4_K_S
                                </option>

                                <option value="ggml_QK5_0">GGML QK5_0</option>
                                <option value="ggml_QK5_1">GGML QK5_1</option>
                                <option value="ggml_QK5_K_M">
                                    GGML QK5_K_M
                                </option>

                                <option value="ggml_Q6_K">GGML Q6_K</option>

                                <option value="ggml_QK8_0">GGML QK8_0</option>
                            </select>
                        </div>
                    </div>

                    <div className="flex">
                        <div>
                            <label className="font-mono text-sm">
                                Context/train seq length?{" "}
                            </label>
                            <TextInput
                                className="w-32 input border text-sm font-mono border-black"
                                setValue={setContextLen}
                                value={contextLen}
                                placeholder="Total Tokens?"
                            />
                        </div>
                        <div className="pl-8">
                            <label className="font-mono text-sm">
                                Batch-size?(only for train){" "}
                            </label>
                            <TextInput
                                className="w-24 input border text-sm font-mono border-black"
                                setValue={setBatchSize}
                                value={batchSize}
                                placeholder="Default 1"
                            />
                        </div>
                        <div className="pl-4">
                            <label className="font-mono text-sm pr-2">
                                GPU?(to be added)
                            </label>
                            <select
                                className="font-mono text-sm border border-gray-500 hover:cursor-not-allowed"
                                name="dropdownGPU"
                                onChange={handleChangeSelection}
                            >
                                <option value="rtx_3090">RTX 3090</option>
                                <option value="rtx_4090">RTX 4090</option>
                                <option value="rtx_4080">RTX 4080</option>
                                <option value="rtx_4060">RTX 4060</option>
                                <option value="rtx_2060">RTX 2060</option>
                                <option value="rtx_2070">RTX 2070</option>
                                <option value="a_6000">A 6000</option>
                            </select>
                        </div>
                    </div>
                    <div>
                        <br></br>
                        {/* <button className='bg-green-50' onClick={handleClick}>Generate Outputs</button> */}
                        <div className="flex">
                            <div className="pr-6">
                                <button
                                    class="font-mono border text-sm border-blue-500 px-4 py-2 bg-blue-100 text-blue-700 hover:bg-blue-200"
                                    onClick={handleClick}
                                >
                                    Find Memory requirement
                                </button>
                            </div>
                            <div>
                                <button
                                    class="font-mono border text-sm border-red-500 px-4 py-2 bg-red-100 text-red-700 hover:cursor-not-allowed"
                                    onClick={handleClickTokS}
                                >
                                    Find ~tokens/s (to be added)
                                </button>
                            </div>
                            <div className="pl-4 pt-1">
                                <button
                                    class="font-mono border text-xs bg-gray-100 border-gray-500  text-black hover:bg-gray-300"
                                    onClick={handleReset}
                                >
                                    Reset
                                </button>
                            </div>
                        </div>
                    </div>
                    <br></br>
                    <hr></hr>
                    <div className="font-bold font-mono">
                        {totalMemoryShown}
                    </div>
                    <div className="font-bold font-mono">{breakDownMemory}</div>
                </div>
                <br></br>
                <br></br>

                <div className="text-xs text-gray-600 font-semibold">
                    PS: These are approximate values & may vary by 500MB-1GB
                    depending on the GPU, model, input, cuda version etc. If
                    your setup has ~1GB over the requirement you should likely
                    be good.
                </div>
                <div>
                    <a
                        className="text-base underline text-blue-600 hover:font-bold"
                        href="https://gist.github.com/RahulSChand/4bc83d1529afc99be14d2a2a54b8e968"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        FAQ
                    </a>
                </div>
                {/* <button>Show Values</button>
      <input type="text" value={output1} readOnly />
      <input type="text" value={output2} readOnly /> */}
            </header>
        </div>
    );
}

export default App;

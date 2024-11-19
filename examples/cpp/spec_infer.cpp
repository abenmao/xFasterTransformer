// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <type_traits>

#include "INIReader.h"
#include "cmdline.h"
#include "sentencepiece_processor.h"
#include "timer.h"
#include "xfastertransformer.h"

extern const char *vocab_opt[];
extern const char *vocab_qwen[];

class TokenizerBase {
public:
    TokenizerBase() {}
    TokenizerBase(std::string &tokenPath) {
        std::filesystem::path filePath(tokenPath);

        if (!(std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath))) {
            std::cout << "[ERROR] " << filePath << " isn't a file or not existed." << std::endl;
            exit(-1);
        }

        const auto status = processor.Load(tokenPath);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
            std::cout << "[ERROR] Fail to load tokenizer file." << std::endl;
            exit(-1);
        }
        vocabSize = processor.GetPieceSize();
    };

    virtual std::vector<int> encode(std::string &input) {
        std::vector<int> output;
        processor.Encode(input, &output);
        addSpecialTokenIds(output);
        return output;
    }

    void addSpecialTokenIds(std::vector<int> &input) {
        input.insert(input.begin(), prefixTokenIds.begin(), prefixTokenIds.end());
        input.insert(input.end(), suffixTokenIds.begin(), suffixTokenIds.end());
    }

    virtual std::string decode(std::vector<int> &ids) {
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }
    virtual std::string decode(int id) { return processor.IdToPiece(id); }

    void printResult(std::vector<int> &ids, int batchSize, int numBeams) {
        if (batchSize * numBeams > 2) {
            printf("[%d]%s [%d]%s ... [%d]%s\n", ids[0], decode(ids[0]).c_str(), ids[1], decode(ids[1]).c_str(),
                    ids[batchSize * numBeams - 1], decode(ids[batchSize * numBeams - 1]).c_str());
        } else if (batchSize * numBeams > 1) {
            printf("[%d]%s [%d]%s\n", ids[0], decode(ids[0]).c_str(), ids[batchSize * numBeams - 1],
                    decode(ids[batchSize * numBeams - 1]).c_str());
        } else {
            printf("[%d]%s ", ids[0], decode(ids[0]).c_str());
        }
    }

    std::vector<std::string> batchDecode(std::vector<int> &input, int batchSize) {
        int seqLen = input.size() / batchSize;
        std::vector<std::string> ret;
        for (int i = 0; i < batchSize; ++i) {
            std::vector<int> tokens(input.begin() + i * seqLen, input.begin() + (i + 1) * seqLen);
            ret.emplace_back(decode(tokens));
        }
        return ret;
    }

protected:
    std::vector<std::string> prefixTokens;
    std::vector<std::string> suffixTokens;
    std::vector<int> prefixTokenIds;
    std::vector<int> suffixTokenIds;

    sentencepiece::SentencePieceProcessor processor;
    int vocabSize;
};

class ChatGLMTokenizer : public TokenizerBase {
public:
    ChatGLMTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        suffixTokens = {"[gMASK]", "<sop>"};
        suffixTokenIds = {processor.PieceToId("[gMASK]"), processor.PieceToId("<sop>")};
    }
};

class ChatGLM2Tokenizer : public TokenizerBase {
public:
    ChatGLM2Tokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        // ChatGLM2's special tokens is not included in sentencepiece. ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        prefixTokens = {"[gMASK]", "sop"};
        prefixTokenIds = {vocabSize + 1, vocabSize + 3};
    }
    std::string decode(std::vector<int> &ids) override {
        ids.erase(std::remove_if(ids.begin(), ids.end(), [this](int value) { return value >= vocabSize; }), ids.end());
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }

    std::string decode(int id) override {
        if (id > vocabSize) {
            return "";
        } else {
            return processor.IdToPiece(id);
        }
    }
};

class LlamaTokenizer : public TokenizerBase {
public:
    LlamaTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) { processor.SetEncodeExtraOptions("bos"); }
};

class BaichuanTokenizer : public TokenizerBase {
public:
    BaichuanTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        // 195: user_id 196: assistant_id
        prefixTokenIds = {195};
        suffixTokenIds = {196};
    }
};

class YaRNLlamaTokenizer : public TokenizerBase {
public:
    YaRNLlamaTokenizer(std::string &tokenPath) { vocabSize = 106963; }

    // TODO: Need to achieve actual encode function
    std::vector<int> encode(std::string &input) override {
        // only for Test
        return std::vector<int>(
                {7454, 2402, 257, 640, 11, 612, 11196, 257, 1310, 2576, 508, 8288, 284, 423, 17545, 13});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }

        std::string text("");
        text.reserve(ids.size());

        for (int id : ids) {
            if (id < vocabSize) {
                if (vocab_list == nullptr)
                    text += "[" + std::to_string(id) + "] ";
                else
                    text += vocab_list[id];
            } else {
                text += "(null) ";
            }
        }

        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = nullptr;
};

class OptTokenizer : public TokenizerBase {
public:
    OptTokenizer(std::string &tokenPath) { vocabSize = 50265; }

    std::vector<int> encode(std::string &input) override {
        return std::vector<int>({2, 11475, 2115, 10, 86, 6, 89, 13412, 10, 410, 1816, 54, 6640, 7, 33, 18848, 4});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }
        std::string text("");
        for (int id : ids) {
            if (id < vocabSize) {
                text += vocab_list[id];
                text += " ";
            } else {
                text += "(null) ";
            }
        }
        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = vocab_opt;
};

class QwenTokenizer : public TokenizerBase {
public:
    QwenTokenizer(std::string &tokenPath) { vocabSize = 151851; }

    // TODO: Need to achieve actual encode function
    std::vector<int> encode(std::string &input) override {
        // only for Test
        return std::vector<int>(
                {12522, 5193, 264, 882, 11, 1052, 24295, 264, 2632, 3743, 879, 14915, 311, 614, 30978, 13});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }

        std::string text("");
        text.reserve(ids.size());

        for (int id : ids) {
            if (id < vocabSize) {
                text += vocab_list[id];
            } else {
                text += "(null) ";
            }
        }

        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = vocab_qwen;
};

class GemmaTokenizer : public TokenizerBase {
public:
    GemmaTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        vocabSize = 256000;
        prefixTokenIds = {2, 2, 106, 1645, 108};
        suffixTokenIds = {107, 108, 106, 2516, 108};
    }
};

TokenizerBase *getTokenizer(std::string &modeltype, std::string &tokenPath) {
    if (modeltype == "gpt") {
        return new OptTokenizer(tokenPath);
    } else if (modeltype == "llama") {
        return new LlamaTokenizer(tokenPath);
    } else if (modeltype == "yarn_llama") {
        return new YaRNLlamaTokenizer(tokenPath);
    } else if (modeltype == "baichuan") {
        return new BaichuanTokenizer(tokenPath);
    } else if (modeltype == "chatglm") {
        return new ChatGLMTokenizer(tokenPath);
    } else if (modeltype == "chatglm2" or modeltype == "chatglm3") {
        return new ChatGLM2Tokenizer(tokenPath);
    } else if (modeltype == "qwen") {
        return new QwenTokenizer(tokenPath);
    } else if (modeltype == "gemma") {
        return new GemmaTokenizer(tokenPath);
    } else {
        std::cout << "[Error] Token list of loaded model is unsupported yet.\n" << std::endl;
        exit(-1);
    }
}

std::map<std::string, xft::DataType> dataTypeMap = {{"fp16", xft::DataType::fp16}, {"bf16", xft::DataType::bf16},
        {"int8", xft::DataType::int8}, {"w8a8", xft::DataType::w8a8}, {"int4", xft::DataType::int4},
        {"nf4", xft::DataType::nf4}, {"bf16_fp16", xft::DataType::bf16_fp16}, {"bf16_int8", xft::DataType::bf16_int8},
        {"bf16_w8a8", xft::DataType::bf16_w8a8}, {"bf16_int4", xft::DataType::bf16_int4},
        {"bf16_nf4", xft::DataType::bf16_nf4}, {"w8a8_int8", xft::DataType::w8a8_int8},
        {"w8a8_int4", xft::DataType::w8a8_int4}, {"w8a8_nf4", xft::DataType::w8a8_nf4}};

std::map<std::string, xft::DataType> KVCacheDataTypeMap
        = {{"fp16", xft::DataType::fp16}, {"int8", xft::DataType::int8}};

std::string getModelType(std::string &modelPath) {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);
    if (reader.ParseError() < 0) {
        printf("[Error] Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();
    return modeltype;
}

void logLastNTokens(TokenizerBase *tokenizer, std::vector<std::vector<int32_t>> tokenIDs, int tokenCount) {
    //return;
    std::vector<std::string> ret;
    for (int i = 0; i < tokenIDs.size(); ++i) {
        std::vector<int> tokens(tokenIDs[i].end() - tokenCount, tokenIDs[i].end());
        ret.emplace_back(tokenizer->decode(tokens));
    }
    std::cout << "'" << ret[0] << "'"<< std::endl;
}

int dynamicAdjustLookahead(int lookaheadK, int acceptedK, int batchSize) {
    static std::vector<int> acceptedKVec;
    acceptedKVec.push_back(acceptedK);

    // 100 refer to flops/byte of the specified hardware
    // 100 for bs16
    int maxK = std::min(8, 100 / batchSize);
    int minK = maxK / 2;

    if (acceptedK <= lookaheadK / 2)
        lookaheadK = std::max(minK, lookaheadK / 2);
    else if (acceptedK == lookaheadK)
        lookaheadK = std::min(maxK, lookaheadK * 2);
    return lookaheadK;
}

int main(int argc, char **argv) {
    cmdline::parser args;

    args.add<std::string>("model", 'm', "path of xft format model", true);
    args.add<std::string>("token", 't', "path of tokenizer", true);
    args.add<std::string>("draft_model", '\0', "path of xft format model of draft", true);
    args.add<std::string>("input", 'i', "input prompt, invalid for Opt model.", false,
            "Once upon a time, there existed a little girl who liked to have adventures.");
    args.add<std::string>("dtype", 'd', "weight data type", false, "fp16");
    //args.add<std::string>("draft_dtype", 'd', "weight data type for draft model", false, "fp16");
    args.add<std::string>("kv_cache_dtype", '\0', "kv cache data type", false, "fp16");
    //args.add<std::string>("draft_kv_cache_dtype", '\0', "kv cache data type for draft model", false, "fp16");
    args.add<int>("input_len", 'l', "input token size", false, -1);
    args.add<int>("output_len", 'o', "max tokens can generate excluded input.", false, 100, cmdline::range(1, 8192));
    args.add<int>("prefix_len", '\0', "shared prefix tokens num.", false, 0);
    args.add<int>("num_beams", 'n', "number of beam size.", false, 1, cmdline::range(1, 32));
    args.add<int>("batch_size", 'b', "batch size.", false, 1, cmdline::range(1, 512));
    args.add<int>("loop", '\0', "number of loop.", false, 1);
    args.add<int>("topK", '\0', "number of highest probability tokens to keep for top-k-filtering.", false, 50);
    args.add<float>("temperature", '\0', "value used to modulate the next token probabilities.", false, 1.0);
    args.add<float>("topP", '\0', "retain minimal tokens above topP threshold.", false, 1.0);
    args.add<float>("repetPen", '\0', "repetition penalty.", false, 1.0);
    args.add("no_stream", '\0', "disable streaming output");
    args.add("do_sample", '\0', "use sampling");
    args.parse_check(argc, argv);

    std::string modelPath = args.get<std::string>("model");
    std::string tokenPath = args.get<std::string>("token");

    bool streamingOutput = !args.exist("no_stream");
    bool doSample = args.exist("do_sample");

    std::string dtype_name = args.get<std::string>("dtype");
    xft::DataType dtype = xft::DataType::fp16;
    std::string kv_cache_dtype_name = args.get<std::string>("kv_cache_dtype");
    xft::DataType KVCacheDataType = xft::DataType::fp16;

    auto it = dataTypeMap.find(dtype_name);
    if (it != dataTypeMap.end()) {
        dtype = it->second;
    } else {
        std::cout << "[Error] Unsupport dtype index: " << dtype_name << std::endl;
        return 0;
    }

    it = KVCacheDataTypeMap.find(kv_cache_dtype_name);
    if (it != KVCacheDataTypeMap.end()) {
        KVCacheDataType = it->second;
    } else {
        std::cout << "[Error] Unsupport KV cache dtype index: " << kv_cache_dtype_name << std::endl;
        return 0;
    }

    int inputSize = args.get<int>("input_len");
    int outputLen = args.get<int>("output_len");
    int prefixLen = args.get<int>("prefix_len");
    int numBeams = args.get<int>("num_beams");
    int batchSize = args.get<int>("batch_size");
    int loop = args.get<int>("loop");
    int topK = args.get<int>("topK");
    float temperature = args.get<float>("temperature");
    float topP = args.get<float>("topP");
    float repetitionPenalty = args.get<float>("repetPen");

    std::string modeltype = getModelType(modelPath);

    auto *tokenizer = getTokenizer(modeltype, tokenPath);
    std::string inputPrompt = args.get<std::string>("input");
    std::vector<int> input = tokenizer->encode(inputPrompt);

    // longer context should be declared in the last, since the bug in xFT
    //xft::AutoModel model(modelPath, dtype, KVCacheDataType);

    // draft model creation
    std::string draftModelPath = args.get<std::string>("draft_model");
    xft::AutoModel draftModel(draftModelPath, dtype, KVCacheDataType);

    // longer context should be declared in the last, since the bug in xFT
    xft::AutoModel model(modelPath, dtype, KVCacheDataType);

    bool isMaster = model.isMaster();

    // Need longer prompt
    if (inputSize > 0 && inputSize > input.size()) {
        input.reserve(inputSize);
        std::vector<int> fakeTokens(inputSize - input.size(), input[2]);
        input.insert(input.begin() + 2, fakeTokens.begin(), fakeTokens.end());
    } else if (inputSize > 0) {
        printf("[Warning] Do not support token size of %d, use %ld instead.\n", inputSize, input.size());
    }

    inputSize = input.size();

    if (isMaster) {
        std::cout << "[INFO] Model path is " << modelPath << std::endl;
        std::cout << "[INFO] DraftModel path is " << draftModelPath << std::endl;
        std::cout << "[INFO] Token path is " << tokenPath << std::endl;
        std::cout << "[INFO] Data type is " << dtype_name << std::endl;
        std::cout << "[INFO] KV cache data type is " << kv_cache_dtype_name << std::endl;
        std::cout << "[INFO] inputSize is " << inputSize << std::endl;
        std::cout << "[INFO] outputLen is " << outputLen << std::endl;
        std::cout << "[INFO] num_beams is " << numBeams << std::endl;
        std::cout << "[INFO] do_samlpe is " << std::boolalpha << doSample << std::endl;
        std::cout << "[INFO] temperature is " << temperature << std::endl;
        std::cout << "[INFO] topK is " << topK << std::endl;
        std::cout << "[INFO] topP is " << topP << std::endl;
        std::cout << "[INFO] repetitionPenalty is " << repetitionPenalty << std::endl;
        std::cout << "[INFO] batch_size is " << batchSize << std::endl;
        std::cout << "[INFO] loop is " << loop << std::endl;
        std::cout << "[INFO] Input prompt is : " << inputPrompt << std::endl;
        std::cout << "[INFO] Input Token Ids is : ";
        for (auto x : input) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    for (int l = 0; l < loop; ++l) {
        std::vector<std::vector<int>> inputIDs, placeholder;
        // request sequence id
        std::vector<int> dseqs, seqs;
        std::vector<int> rejectedK;
        std::vector<int32_t> rectInputId;

        // lookahead init value
        int lookahead_k = 8;

        SearcherConfig config;
        config.maxLen = input.size() + outputLen + lookahead_k * 2;

        Timer spec(true, "Spec infer");

        for (int n = 0; n < batchSize; ++n)
            inputIDs.push_back(input);
        seqs = model.set_input(inputIDs, seqs, config);
        // validation prefill
        auto res = model.validateBatch(0, placeholder);
        auto firstId = std::get<0>(res);
        for (int n = 0; n < inputIDs.size(); ++n)
            inputIDs[n].push_back(firstId[n]);

        Timer gen(true, "Gen-only infer");
        int tokenCount = 1;
        // just for the first seq [0], TODO: mengchen
        while (tokenCount < outputLen) {
            lookahead_k = std::min(lookahead_k, outputLen - tokenCount);
            {
            Timer lookahead(true, "lookahead N");

            // create new workingGroup(with kvCaches) for the lookahead task thread
            if (dseqs.size() == 0)
                dseqs = draftModel.set_input(inputIDs, dseqs, config);

	    // inferencing
            auto proposals = draftModel.lookaheadN(lookahead_k, rejectedK, rectInputId);

	    // prepare inputIDs for validate
            // if validation has finished prefill
            if (seqs.size() > 0)
                inputIDs.clear();
            for (int i = 0; i < proposals.size(); ++i) {
                // if validation has finished prefill
                if (seqs.size() > 0)
                    inputIDs.push_back(proposals[i]);
                else
                    inputIDs[i].insert(inputIDs[i].end(), proposals[i].begin(), proposals[i].end());
            }
            }

            // logging
            logLastNTokens(tokenizer, draftModel.getGeneratedTokens(), lookahead_k);

            {
            Timer validate(true, "validate batch");

            // create new workingGroup(with kvCaches) for the validation task thread
            if (seqs.size() == 0) {
                seqs = model.set_input(inputIDs, seqs, config);
                inputIDs.clear();
            }
            // validating
            auto res = model.validateBatch(lookahead_k, inputIDs);
            // prepare the rectified tokens for lookahead
            rectInputId = std::get<0>(res);
            rejectedK = std::get<1>(res);
            // update gen token count, just for the first seq [0], TODO: mengchen
            tokenCount += lookahead_k - rejectedK[0] + 1;
            }

            // logging
            for (int n=0; n<rectInputId.size(); ++n) {
                std::cout << "rejected " << rejectedK[n] << " rect " << rectInputId[n] << std::endl;
            }
            // just for the first seq [0], TODO: mengchen
            logLastNTokens(tokenizer, model.getGeneratedTokens(), lookahead_k - rejectedK[0] + 1);

            // adjust lookahead_k
            lookahead_k = dynamicAdjustLookahead(lookahead_k, lookahead_k - rejectedK[0], seqs.size());
        }

        // logging
        logLastNTokens(tokenizer, model.getGeneratedTokens(), tokenCount);
        std::cout << "gen tokens " << tokenCount << std::endl;
    }

    return 0;
}

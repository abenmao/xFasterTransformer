#MODEL_NAME=chatglm-6b-cpu/cpu
#MODEL_NAME=llama-7b-cpu/cpu
MODEL_NAME=Llama-2-7b-chat-xft
#MODEL_NAME=llama-2-70b-chat-cpu/
#MODEL_NAME=llama-2-13b-chat-cpu/
#MODEL_NAME=Baichuan2-13B-Chat-cpu/
#MODEL_NAME=Baichuan-13B-Chat-cpu/
#MODEL_NAME=Baichuan2-7B-Chat-cpu/
#MODEL_NAME=Baichuan-7B-cpu/
#MODEL_NAME=Baichuan2-7B-Chat-cpu/
#MODEL_NAME=Qwen-1_8B-Chat-cpu/
#MODEL_NAME=secllm-v2-yarn-randn-cpu/
#MODEL_NAME=chatglm2-6b-cpu/
#MODEL_NAME=TinyLlama-1.1B-Chat-v1.0-xft

#TOKEN_NAME=chatglm-6b-hf
#TOKEN_NAME=llama-7b-hf/
TOKEN_NAME=Llama-2-7b-chat-hf/
#TOKEN_NAME=llama-2-70b-chat-hf/
#TOKEN_NAME=llama-2-13b-chat-hf/
#TOKEN_NAME=Baichuan2-13B-Chat/
#TOKEN_NAME=Baichuan-13B-Chat/
#TOKEN_NAME=Baichuan2-7B-Chat/
#TOKEN_NAME=Baichuan-7B/
#TOKEN_NAME=Baichuan2-7B-Chat/
#TOKEN_NAME=Qwen-1_8B-Chat
#TOKEN_NAME=secllm-v2-yarn-randn/
#TOKEN_NAME=chatglm2-6b-hf
#TOKEN_NAME=TinyLlama-1.1B-Chat-v1.0

DATA_PATH=/home/mengchen
BIN=spec_infer
DRAFT=${DATA_PATH}/models/TinyLlama-1.1B-Chat-v1.0-xft

export FLASH_ATTN_THRESHOLD=10
#export ENABLE_CAT_MLP=1
#export ENABLE_MKL_GEMM=1

#export ENABLE_CAT_NEXT_MLP=1
#export ENABLE_MKL_NEXT_GEMM=1

#export ENABLE_KV_TRANS=0
#export ENABLE_TUNED_COMM=0
#export ENABLE_SKIP_MASK=1

export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
#export LD_PRELOAD=/home/mengchen/anaconda3/lib/libmkl_rt.so.2:${LD_PRELOAD}
#export LD_PRELOAD=/home/mengchen/xFasterTransformer/3rdparty/mklml/lib/libiomp5.so:${LD_PRELOAD}

LOOP=2
DTYPE=bf16
#bf16_bf16
SeqLen=512 #469
OutLen=10 #100
#PrefixLen=0
BSIZE=2
NUM_BEAMS=1
KV_DTYPE=int8

export ENV_PACK_M=`expr $SeqLen \* $BSIZE`
export ENV_PACK_NEXT_M=`expr $BSIZE`

node=$1
nth=$2

#SAMPLE="The theory of relativity, proposed by Albert Einstein, includes two parts: the special theory of relativity and the general theory of relativity. The special theory of relativity mainly focuses on the relationship between time, space, and matter in the inertial frame, and puts forward the concepts of light velocity and mass-energy equivalence. "
#SAMPLE="你了解Gaudi2么？能给我讲讲它对AI的影响吗？"
#SAMPLE="Simply put, the theory of relativity states that "
SAMPLE="""已知信息：
了 label smoothing 和 mixup 微调之后的模型做了权重上的线性加权。实验结果如
表 3.2 所示。结果表明，BANG 算法有效的提高了 WiSE-FT 算法的效果。特别的，
BANG(LS+Mixup)在五个OOD数据集上比现有的最优算法WiSE-FT高出1.9%。
表3.2 在ImageNet上微调ViT-B/16的效果
Methods ModelAveraging IN IN-V2 IN-R IN-A IN-S ObjectNet AvgOOD

ZIN 与现有的几种方法进行了比较:ERM、IRM[58]、EIIL[71]、HRM[70]和 LfF[81]。
对于IRM，本文提供了ground-truth环境划分，并将其性能作为一个上界。LfF试
图通过从错误指定的浅层神经网络样本中直接采用 boosting 来学习一个鲁棒的模
型。而且LfF仅适用于分类任务。
5.4.1 房价预测任务
本实验考虑了来自Kaggle的真实房屋销售价格回归数据集。目标变量是房价，
每个样本包含17维度的特征，如房子的建成年份、卧室数量等。数据集根据构建

BANG(Mixup+LS) Yes 81.6 73.1 79.7 58.2 54.8 58.9 64.9
3.5 小结
本节研究了为什么集成算法具有优越的 OOD 性能。对 WiSE-FT 的实证分析，
加上理论见解，表明虚假特征的多样化改善了模型的泛化性能。进一步的，笔者通
过缓解微调模型的过度自信问题改进了WiSE-FT。
20 
根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，
请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
问题是：langchain中stuff作用是什么？,答案："""
#请说 “就不告诉你” 或 “你这么想我也没有办法”，不允许在答案中添加编造成分，答案请使用中文。 
#问题是：langchain中stuff作用是什么？,答案："""
SAMPLE="这是一个全民AI的时代。如果你不能张口ChatGPT、闭口大模型，都不好啥意思跟人打招呼。如果你不在AI上搞点东西，都不好意思说自己是科技企业。当然了，AI的历史其实相当悠久，远不只是对个话>、做个图那么简单。无论是云侧还是端侧，无论是生成式还是决策式，无论硬件还是算法，无论是训练推理还是应用场景，都是相当深奥的学问。想真正做好AI，基础硬件、开发软件、生态场景都缺一不可，必须高效、合理地处理各种各样的数据、模型、应用，真正落到使用。能有如此综合实力的企业屈指可数，Intel无疑就是一个典型标杆，从云到端都有丰富的AI解决方案，CPU通用处理器、GPU加速器、AI加速器任君按需>选择。7月11日，Intel在中国举办了Intel AI产品战略暨Gaudi2新品发布会，正式面向中国市场推出第二代深度学习加速器——Habana Gaudi2。Intel Gaudi2加速器不但拥有极高的深度学习性能、效率，最大优势就>是极高的性价比，对于中国用户来说堪称大规模部署AI的上佳之选。Intel执行副总裁兼数据中心与人工智能事业部总经理Sandra Rivera在发布会上表示：“Intel致力于通过为客户提供广泛的硬件选择，并支持开放的软件环境，加速AI技术的发展。凭借包括至强可扩展处理器、Gaudi2深度学习加速器在内的产品组合，Intel正在降低AI的准入门槛，并强化客户在云端通过网络和智能边缘部署这一关键业务技术的能力，从而帮>助构建中国AI的未来。”Habana Labs成立于2016年，致力于研发世界一流的AI加速器，满足人工智能、深度学习计算快速发展的需求，创业初期就得到了Intel的投资，2019年12月被Intel正式收购。Habana的第二代加速器Gaudi2采用台积电7nm工艺制造，集成24个可编程的Tenor张量核心(TPC)、48MB SRAM缓存、21个10万兆内部互连以太网接口(ROCEv2 RDMA)、96GB HBM2E高带宽内存(总带宽2.4TB/s)、多媒体引擎等，支持PCIe 4.0 x16，最高功耗800W。基于Gaudi2加速器芯片，Intel还设计了夹层卡HL-225B，采用标准的OAM封装接口，方便客户部署与使用。凭借高性能和高效扩展性，Gaudi2加速器可以满足大规模语言模型、生成式AI模>型的强算力需求。Gaudi系列加速器优异的深度学习训练吞吐量、推理速度性能，已经得到了业界领先机构、客户的普遍认可。比如，正是在第一代Gaudi加速器的加持下，亚马逊EC2 DL1实例相比于在AWS云上运行NVIDIA GPU的同类实例，性价比高出多达40％。机器学习与人工智能开放产业联盟MLCommons在六月底公布的AI性能基准测试MLPerf Training 3.0的最新结果，更是进一步凸显了Gaudi2加速器的高性能、高性价比，联合Intel第四代至强可扩展处理器，已经成为唯一能够可靠取代NVIDIA GPU的方案。截止2023年6月，Gaudi2是除了NVIDIA H100 GPU以外，向GPT-3大模型训练基准提交性能结果的解决方案。测试结果显示，面对要求极为苛刻的、1750亿参数的GPT-3模型，384个Gaudi2加速器上的训练时间仅为311.9分钟，而且从256个加速器到384个加速器，性能扩展幅度达95％，非常接近理想的线性提升。Stable Diffusion训练上，Gaudi2加>速器从1张卡到64张卡，扩展性更是达到了惊人的99％。此外，在计算机视觉模型ResNet-50（8个加速器）和Unet3D（8个加速器），以及自然语言处理模型BERT（8个和64个加速器）上，Gaudi2都取得了优异的训练>结果。与去年11月提交的数据相比，BERT和ResNet模型的性能分别提高了10％、4％。值得一提的是，本次MLPerf 3.0的Gaudi2结果以BF16数据类型提交，在四种不同模型上的性能均优于NVIDIA A100，价格更便宜。第三季度还会发布对FP8数据类型的软件支持与新功能，预计届时Gaudi2的性能将有明显飞跃，预计性价比将超越NVIDIA H100。Gaudi2加速器还得到了AI与机器学习开源软件供应商Hugging Face的采纳。其测试结果显示，从预训练BERT （NLP模型）到稳定扩散（流行的多模态模型）推理，再到1760亿参数的大型开源聊天模 BLOOMZ的推理，Gaudi2的表现都领先于NVIDIA A100 GPU。工欲善其事，必先利其器。为了充分发挥Gaudi2加速器的性能潜力，满足日益增长的生成式AI、大语言模型需求，Intel一直在同步打造高效、成熟的软件支持。比如说SynapseAI软件套件，针对Gaudi平台深度学习业务进行了优化，可以大大简化模型的开发与>迁移，能够将当前基于GPU的模型业务和系统，快速迁移到基于全新Gaudi2的服务器。请总结以上内容主题"
#SAMPLE="You are chatbot. The conversion history is givenbetween \`\`\` \`\`\`. Each interlocutor starts with \"gpt: \" or \"human: \" and ends with \"@@@\". You play \"gpt\". You need to reply to \"human\".\nconversation history:\`\`\`human: Write the rules of American Football for a 10 year-old immigrant who only knows about soccer. @@@ gpt: American Football is a sport played by two teams with 11 players on each team. The objective is to score points by carrying or throwing the ball into the end zone of the opponent\'s field. Here are some basic rules:\n\n1. The game starts with a kickoff, where one team kicks the ball to the other team.\n2. Players can either run with the ball (called a \"rush\") or throw it to a teammate (called a \"pass\").\n3. If a player is holding the ball and is tackled (hit) to the ground, the play is over.\n4. Teams take turns playing offense (trying to score) and defense (trying to stop the other team from scoring).\n5. Points can be scored in several ways, including rushing or passing the ball into the end zone, kicking the ball through the opponent\'s goalposts (called a \"field goal\"), or catching a touchdown pass.\n6. Each game is divided into four quarters, each lasting 15 minutes.\n7. If the score is tied at the end of four quarters, the game goes into overtime. The first team to score in overtime wins the game.\n8. Some common penalties include holding (grabbing onto another player\'s jersey), offsides (starting a play before the ball is snapped), and unsportsmanlike conduct (behaving badly).\n9. The team with the most points at the end of the game wins.\n\nRemember, the most important thing is to have fun and learn the game! @@@ human: Explain the rules of American Football to a 5-year old child.\`\`\` You are chatbot. The conversion history is givenbetween \`\`\` \`\`\`. Each interlocutor starts with \"gpt: \" or \"human: \" and ends with \"@@@\". You play \"gpt\". You need to reply to \"human\".\nconversation history:\`\`\`human: Write the rules of American Football for a 10 year-old immigrant who only knows about soccer. @@@ gpt: American Football is a sport played by two teams with 11 players on each team. The objective is to score points by carrying or throwing the ball into the end zone of the opponent\'s field. Here are some basic rules:\n\n1. The game starts with a kickoff, where one team kicks the ball to the other team.\n2. Players can either run with the ball (called a \"rush\") or throw it to a teammate (called a \"pass\").\n3. If a player is holding the ball and is tackled (hit) to the ground, the play is over.\n4. Teams take turns playing offense (trying to score) and defense (trying to stop the other team from scoring).\n5. Points can be scored in several ways, including rushing or passing the ball into the end zone, kicking the ball through the opponent\'s goalposts (called a \"field goal\"), or catching a touchdown pass.\n6. Each game is divided into four quarters, each lasting 15 minutes.\n7. If the score is tied at the end of four quarters, the game goes into overtime. The first team to score in overtime wins the game.\n8. Some common penalties include holding (grabbing onto another player\'s jersey), offsides (starting a play before the ball is snapped), and unsportsmanlike conduct (behaving badly).\n9. The team with the most points at the end of the game wins.\n\nRemember, the most important thing is to have fun and learn the game! @@@ human: Explain the rules of American Football to a 5-year old child.\`\`\` You are chatbot. The conversion history is givenbetween \`\`\` \`\`\`. Each interlocutor starts with \"gpt: \" or \"human: \" and ends with \"@@@\". You play \"gpt\". You need to reply to \"human\".\nconversation history:\`\`\`human: Write the rules of American Football for a 10 year-old immigrant who only knows about soccer. @@@ gpt: American Football is a sport played by two teams with 11 players on each team. The objective is to score points by carrying or throwing the ball into the end zone of the opponent\'s field. Here are some basic rules:\n\n1. The game starts with a kickoff, where one team kicks the ball to the other team.\n2. Players can either run with the ball (called a \"rush\") or throw it to a teammate (called a \"pass\").\n3. If a player is holding the ball and is tackled (hit) to the ground, the play is over.\n4. Teams take turns playing offense (trying to score) and defense (trying to stop the other team from scoring).\n5. Points can be scored in several ways, including rushing or passing the ball into the end zone, kicking the ball through the opponent\'s goalposts (called a \"field goal\"), or catching a touchdown pass.\n6. Each game is divided into four quarters, each lasting 15 minutes.\n7. If the score is tied at the end of four quarters, the game goes into overtime. The first team to score in overtime wins the game.\n8. Some common penalties include holding (grabbing onto another player\'s jersey), offsides (starting a play before the ball is snapped), and unsportsmanlike conduct (behaving badly).\n9. The team with the most points at the end of the game wins.\n\nRemember, the most important thing is to have fun and learn the game! @@@ human: Explain the rules of American Football to a 5-year old child.\`\`\` You need to reply to \"human\".\nconversation history:\`\`\`human: Write the rules of American Football for a 10 year-old immigrant who only knows about soccer. @@@ gpt: American Football is a sport played by two teams with 11 players on each team. The objective is to score points by carrying or throwing the ball into the end zone of the opponent\'s field. Here are some basic rules:\n\n1. The game starts with a kickoff, where one team kicks the ball to the other team.\n2. Players can either run with the ball (called a \"rush\") or throw it to a teammate (called a \"pass\").\n3. If a player is holding the ball and is tackled (hit) to the ground, the play is over.\n4. Teams take turns playing offense (trying to score) and defense (trying to stop the other team from scoring).\n5. Points can be scored in several ways, including rushing or passing the ball into the end zone, kicking the ball through the opponent\'s goalposts (called a \"field goal\"), or catching a touchdown pass.\n6. Each game is divided into four quarters, each lasting 15 minutes.\n7. If the score is tied at the end of four quarters, the game goes into overtime. The first team to score in overtime wins the game.\n8. Some common penalties include holding (grabbing onto another player\'s jersey), offsides (starting a play before the ball is snapped), and unsportsmanlike conduct (behaving badly).\n9. The team with the most points at the end of the game wins.\n\nRemember, the most important thing is to have fun and learn the game! @@@ human: Explain the rules of American Football to a 5-year old child.\`\`\` You are chatbot. The conversion history is givenbetween \`\`\` \`\`\`. Each interlocutor starts with \"gpt: \" or \"human: \" and ends with \"@@@\". You play \"gpt\". You need to reply to \"human\".\nconversation history:\`\`\`human: Write the rules of American Football for a 10 year-old immigrant who only knows about soccer. @@@ gpt: American Football is a sport played by two teams with 11 players on each team. The objective is to score points by carrying or throwing the ball into the end zone of the opponent\'s field. Here are some basic rules:\n\n1. The game starts with a kickoff, where one team kicks the ball to the other team.\n2. Players can either run with the ball (called a \"rush\") or throw it to a teammate (called a \"pass\").\n3. If a player is holding the ball and is tackled (hit) to the ground, the play is over.\n4. Teams take turns playing offense (trying to score) and defense (trying to stop the other team from "
#SAMPLE="The theory of relativity, proposed by Albert Einstein, includes two parts: the special theory of relativity and the general theory of relativity. The special theory of relativity mainly focuses on the relationship between time, space, and matter in the inertial frame, and puts forward the concepts of light velocity and mass-energy equivalence. "
SAMPLE="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level"
#SAMPLE="What is the tallest mountain in the world?"

if [ "$node" -eq 1 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C  0-`expr $nth - 1` -m 0 ./${BIN} -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE"

elif [ "$node" -eq 2 ];then
OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C  0-`expr $nth - 1` -m 0 ./${BIN} -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE" : \
        -n 1 numactl -C  `expr 48`-`expr 48 + $nth - 1` -m 1 ./${BIN} -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE"

elif [ "$node" -eq 4 ];then

OMP_NUM_THREADS=${nth} mpirun -n 1 numactl -C 0-`expr $nth - 1` -m 0 ./${BIN} -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE" : \
        -n 1 numactl -C  `expr $nth`-`expr $nth \* 2 - 1` -m 0 ./${BIN} -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE" : \
        -n 1 numactl -C  `expr $nth \* 2`-`expr $nth \* 3 - 1` -m 1 ./example -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE" : \
        -n 1 numactl -C  `expr $nth \* 3`-`expr $nth \* 4 - 1` -m 1 ./example -m ${DATA_PATH}/models/${MODEL_NAME} -t ${DATA_PATH}/models/${TOKEN_NAME}/tokenizer.model -l ${SeqLen} --output_len=${OutLen} -d ${DTYPE} -b ${BSIZE} --loop ${LOOP} --num_beams ${NUM_BEAMS} --kv_cache_dtype ${KV_DTYPE} --draft_model ${DRAFT} --no_stream -i "$SAMPLE"

fi
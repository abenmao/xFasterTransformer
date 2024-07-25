import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

model_path = "/home/mengchen/models/chatglm3-6b-hf"
model_path = "/home/mengchen/models/Llama-2-7b-chat-hf"
model_path = "/home/mengchen/models/Baichuan-13B-Chat"
model_path = "/home/mengchen/models/Baichuan2-13B-Chat"
model_path = "/home/mengchen/models/Qwen2.5-14B-Instruct"

import pdb
pdb.set_trace()

device = torch.device('cpu:0')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device=device)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().to(device)
#model = model.eval()
input1 = "你是一个医学专家，擅长回答用户的医疗问题，请你根据参考内容和要求回答问题。\n\n参考内容：\n\"\"\"\n问题：干细胞对糖尿病的治疗作用\n回答：干细胞对治疗糖尿病有一定作用，但技术并不是十分成熟，仍在科学研究阶段。干细胞是具有自我增殖、分化、修复功能的原始细胞，理论上干细胞可以帮助恢复已经受损的胰岛功能，纠正糖尿病患者体内胰岛细胞的紊乱状态，有利于血糖控制。干细胞能够诱导自身胰岛素分泌干细胞的增殖分化，或者刺激机体自身具有胰岛素分泌功能的β细胞功能，使其胰岛素的分泌量增加，提高机体自身胰岛素分泌水平，并降低胰岛素抵抗，使糖尿病患者尽量脱离药物治疗。另外，干细胞可以分泌大量生长因子，改善肾脏局部微环境、促进血管生成，改善糖尿病肾病引起的病理性改变。但目前干细胞对糖尿病的治疗效果并不确切，仍在实验室试验和临床试验中。糖尿病的典型治疗方法是五驾马车法，包括饮食控制、运动治疗、药物治疗、监测血糖以及糖尿病宣传教育等，药物治疗需在医生的指导下应用，常用降糖药物包括促胰岛素分泌剂，如格列本脲、格列吡嗪，还有二甲双胍以及SGLT-2抑制剂，如达格列净等。此外，糖尿病患者在日常生活中要遵照医生制定的饮食计划规律进食，定时定量，避免暴饮暴食，多进行运动锻炼，以有氧运动为主，如快走、骑自行车等，遵医嘱定期前往医院复。\n\n\"\"\"\n\n要求：\n1. 判断用户问题能否在参考内容中找到回答，如果可以，请输出该问题的回答\n2. 如果用户问题无法在参考内容中找到回答，请输出“很抱歉，关于您的问题，我还在学习中，暂时无法给出回答。”\n\n问题：\n干细胞治疗糖尿病新进展"
input2 = "#角色\n你是一位资深、专业、有温度的药剂师，擅长解答药品相关问题\n\n#任务\n依据下面药品说明书中的药品资料，每个证据有自己的序号，按照指令最后的要求回答问题\n\n#药品说明书\n\"\"\"\n[1]: “【阿司匹林片】通用名称：阿司匹林片\t英文名称：Aspirin Tablets\t汉语拼音：A Si Pi Lin Pian\t成份：本品每片含主要成份阿司匹林0.3克。辅料为淀粉。\t所属类别：化药及生物制品 >> 消化道及代谢类药物 >> 口腔病药物 >> 口腔病药物\n化药及生物制品 >> 血液和造血器官用药 >> 抗血栓形成药 >> 抗血栓形成药\n化药及生物制品 >> 神经系统用药 >> 镇痛药 >> 其它解热镇痛药\t性状：本品为白色片。\t适应症：用于普通感冒或流行性感冒引起的发热，也用于缓解轻至中度疼痛如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。\t规格：0.3g*100s\t用法用量：成人：一次1片，若发热或疼痛持续不缓解，可间隔4-6小时重复用药一次。24小时内不超过4片。儿童用量请咨询医师或药师。\t不良反应：1.较常见的有恶心、呕吐、上腹部不适或疼痛等胃肠道反应。2.较少见或罕见的有（1）胃肠道出血或溃疡，表现为血性或柏油样便，胃部剧痛或呕吐血性或咖啡样物，多见于大剂量服药患者。（2）支气管痉挛性过敏反应，表现为呼吸困难或哮喘。（3）皮肤过敏反应，表现为皮疹、荨麻疹、皮肤瘙痒等。（4）血尿、眩晕和肝脏损害。\t禁忌：1.妊娠期、哺乳期妇女禁用。2.哮喘，鼻息肉综合征，对阿司匹林及对其他解热镇痛药过敏者禁用。3.血友病或血小板减少症，溃疡病活动期的患者禁用。4.服用本品期间禁止饮酒。\t注意事项：1.本品为对症治疗药，用于解热连续使用不超过3天，用于止痛不超过5天，症状未缓解请咨询医师或药师。2.不能同时服用其他含有解热镇痛药的药品（如某些复方抗感冒药）。3.年老体弱患者应在医师指导下使用。4.服用本品期间不得饮酒或含有酒精的饮料。5.痛风、肝肾功能减退、心功能不全、鼻出血、月经过多以及有溶血性贫血史的患者慎用。6.发热伴脱水的患儿慎用。7.如服用过量或出现严重不良反应，应立即就医。8.对本品过敏者禁用，过敏体质者慎用。9.本品性状发生改变时禁止使用。10.请将本品放在儿童不能接触的地方。11.儿童必须在成人监护下使用。12.如正在使用其他药品，使用本品前请咨询医师或药师。\t孕妇及哺乳期妇女用药：本品易于通过胎盘，并可由乳汁分泌，故妊娠期、哺乳期妇女禁用。\t儿童用药：发热伴脱水的患儿慎用。\t老年用药：本品未进行该项实验且无可靠参考文献。\t药物相互作用：1.不应与含有本品的同类制剂及其他解热镇痛药同用。2.本品不宜与抗凝血药（如双香豆素、肝素）及溶栓药（链激酶）同用。3.抗酸药如碳酸氢钠等可增加本品自尿中的排泄，使血药浓度下降，不宜同用。4.本品与糖皮质激素（如地塞米松等）同用，可增加胃肠道不良反应。5.本品可加强口服降糖药及甲氨蝶呤的作用，不应同用。6.如正在服用其他药品，在服用本品前请咨询医师或药师。\t药理作用：本品抑制前列腺素合成，具有解热、镇痛和抗炎作用。\t贮藏：密封，置阴凉干燥处。\t包装：塑料瓶装，0.3克/片，100片/瓶。\t有效期：24个月\t批准文号：国药准字H44020656\t生产企业：广东百澳药业有限公司【企业名称】广东百澳药业有限公司【地址】开平市沙塘镇工业开发区【省份/国家】广东省\t妊娠分级：C；D-如在妊娠晚期大量使用\t哺乳期分级：阿司匹林: L3 半衰期2.5-7h极少量可分泌入乳汁，不良反应的报道较少。可诱发瑞氏综合征，不建议选用。\t”\n\"\"\"\n\n#要求\n1、基于【药品说明书】进行回答，如果问题涉及超过【药品说明书】范围的知识，请结合你的认知作答\n2、回答先给解释，再给结论。\n3、如果回答中的句子来自于某个证据段落，需要在该句子结尾引用证据的序号，引用格式示例如下：[1]，[3]，[2][5]。\n4、请在回答中进行安全风险提示，比如”请立即报告医生“，”请及时就医“等话术。\n5、回答需要清晰简洁，不要使用第一人称或第三人称回答，不要出现「根据药品说明书」、「基于文档」等描述，回答中需要包含风险提示语句，比如“需要在医生或药师指导下使用药品。”，回答字数不超过100字。\n6、判断问题是否是问药物禁忌的，如果是，在回答中加上“对药物成分过敏者禁用”。\n\n阿司匹林抑制中枢PG合成中的PG是什么？\n"
input3 = "It's done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise"

import json
with open("./params.json", "r", encoding='utf-8') as file:
    data = json.load(file)
input4 = data['messages'][0]['content']

input5 = "你好，给我讲一个故事，大概10个字"
input6 = "过敏史: 无青霉素过敏史，其它药物过敏：否认         重要药物应用史：见现病史。\t\n既往史: 疾病史：糖尿病5年，口服二甲双胍控制好。56年前因血吸虫病在当地医院治疗，无高血压史        传染病史：否认肝炎史、结核史   手术史外伤史：5年前胆囊结石进腹手术切除，否认外伤史\t\n\n你是一名经验丰富的临床医生，具备非常丰富的医学知识，非常擅长根据{病历内容}进行医疗实体抽取。\n*任务\n请基于上面的{病历内容}，抽取其中阳性的过敏史、疾病、手术\n*要求\n1.抽取结果需要来源于病历，不抽取已否认或阴性的过敏史、疾病和手术，不要遗漏阳性传染疾病；\n2.抽取结果中疾病只保留：高血压、糖尿病、冠心病、脑梗、脑出血、心肌梗死、肝硬化、国家法定传染病；\n3.抽取的结果使用#号进行分割，不要返回json格式；\n4.输出内容的格式需要严格与如下例子保持一致：\n    \"\"\"\n    过敏：xx1过敏#xx2过敏\n    疾病：xx1#xx2#xx3\n    手术：xx1#xx2\n    \"\"\"\n*示例\n1.\n输入：\n    疾病史：糖尿病5年，口服二甲双胍控制好。56年前因血吸虫病在当地医院治疗，无高血压史。  传染病史：否认肝炎史、结核史。 手术史外伤史：5年前胆囊结石进腹手术切除，否认外伤史。\n输出:\n    过敏：无\n    疾病：糖尿病#血吸虫病\n    手术：胆囊结石进腹切除手术"
input7 = "介绍周杰伦的主要作品"

input = input7
pdb.set_trace()
#inputs = tokenizer.build_chat_input(input, history=[], role="user").input_ids
inputs = tokenizer(input, return_tensors="pt", padding=False).input_ids
suffix = torch.tensor([[196]])
prefix = torch.tensor([[195]])
inputs = torch.cat((prefix, inputs, suffix), dim=1)
#eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
#                tokenizer.get_command("<|observation|>")]
outputs = model.generate(inputs, max_new_tokens=200, do_sample=False, temperature=0.8, top_k=50, top_p=0.8)#, eos_token_id=eos_token_id)
print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

response, history = model.chat(tokenizer, input, max_length=2000, do_sample=False, temperature=0.8, top_k=50, top_p=0.8, history=[])
print(response)

import argparse
import random
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'

        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe
        ))

        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'MiniMind model parameter quantity : {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain model's solitaire ability （ unable to talk ）
        prompt_datas = [
            ' the basic principles of marxism ',
            ' the main functions of the human brain ',
            ' the principle of gravity is ',
            ' the highest mountain in the world is ',
            ' carbon dioxide in the air ',
            ' the largest animal on earth is ',
            ' the food in hangzhou is '
        ]
    else:
        if args.lora_name == 'None':
            #  general conversation issues
            prompt_datas = [
                ' please introduce yourself 。',
                ' which subject are you better at ？',
                " lu xun's 《 madman's diary 》 how to criticize feudal ethics ？",
                ' my cough has been going on for two weeks ， do you need to go to the hospital for examination? ？',
                ' a detailed introduction to the physical concept of the speed of light 。',
                ' recommend some special foods in hangzhou 。',
                ' Please explain the “Large Language Model” to me this concept 。',
                ' how to understand ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            #  area-specific issues
            lora_prompt_datas = {
                'lora_identity': [
                    " who are you ChatGPT bar 。",
                    " may i have your name ？",
                    " you and openai what's the relationship ？"
                ],
                'lora_medical': [
                    " i've been feeling dizzy lately ， what might be the reason ？",
                    ' my cough has been going on for two weeks ， do you need to go to the hospital for examination? ？',
                    ' what should i pay attention to when taking antibiotics ？',
                    ' the physical examination report shows that the cholesterol is high ， what do i do ？',
                    ' what should pregnant women pay attention to in their diet ？',
                    ' how to prevent osteoporosis in the elderly ？',
                    " i've always felt anxious lately ， how to alleviate ？",
                    ' if someone faints suddenly ， how to first rescue ？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


#  set reproducible random seeds
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    #  here max_seq_len（ maximum output length ） does not mean that the model has the corresponding long text performance ， prevent only QA there is a problem of being truncated
    # MiniMind2-moe (145M)：(hidden_size=640, num_hidden_layers=8, use_moe=True)
    # MiniMind2-Small (26M)：(hidden_size=512, num_hidden_layers=8)
    # MiniMind2 (104M)：(hidden_size=768, num_hidden_layers=16)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    #  carry historical dialogue context number
    # history_cnt need to be set to even number ， right now 【 user problems ,  model answer 】 for 1 group ； set as 0 hour ， that is the current query don't carry historical texts
    #  when the model has not been extrapolated and fine-tuned ， in longer context chat_template it is inevitable that the performance will be significantly degraded ， therefore, you need to pay attention to the settings here
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--load', default=0, type=int, help="0:  native torch weight ，1: transformers load ")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0:  pre-trained model ，1: SFT-Chat model ，2: RLHF-Chat model ，3: Reason model ，4: RLAIF-Chat model ")
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0]  automatic testing \n[1]  manual input \n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  #  if you need to fix each output, change it to 【 fixed 】 random seeds of
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')


from collections import Counter

def get_reference_vocab(references, min_count=3):
    counter = Counter()
    for sent in references:
        for word in sent.lower().split():
            counter[word] += 1
    # 只保留高频词（比如至少出现3次）
    return set([w for w, c in counter.items() if c >= min_count])


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name().replace('_', ' '))
    return synonyms

def replace_with_common_synonym(sentence, high_freq_words):
    words = sentence.split()
    new_words = []
    for w in words:
        if w.lower() not in high_freq_words:
            # 查找同义词，并优先替换成高频词表中的词
            candidates = get_synonyms(w)
            common_syn = [c for c in candidates if c in high_freq_words]
            if common_syn:
                new_words.append(common_syn[0])  # 替换为第一个高频同义词
            else:
                new_words.append(w)
        else:
            new_words.append(w)
    return " ".join(new_words)



def dedup_sentence_words(sentence):
    words = sentence.strip().split()
    deduped = []
    prev = None
    for w in words:
        if w != prev:
            deduped.append(w)
        prev = w
    return ' '.join(deduped)


def dedup_all_sentence_words(sentence):
    words = sentence.strip().split()
    seen = set()
    deduped = []
    for w in words:
        if w not in seen:
            deduped.append(w)
            seen.add(w)
    return ' '.join(deduped)


def dedup_words_in_captions(caption_list, mode='consecutive'):
    """
    mode: 'consecutive' 只去连续重复; 'all' 去所有重复
    """
    if mode == 'consecutive':
        func = dedup_sentence_words
    else:
        func = dedup_all_sentence_words
    return [func(s) for s in caption_list]


from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载TinyLlama-1.1B-Chat模型（或换phi-2、Qwen1.8B-Chat等）
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def polish_english(sentence, max_new_tokens=40):
    # 指令prompt，专为英文错句润色
    prompt = (
        "Please correct and rewrite the following sentence in fluent, grammatically correct English:\n"
        f"{sentence}\n"
        "Corrected:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 可做后处理只提取Corrected:后面的内容
    if "Corrected:" in result:
        result = result.split("Corrected:")[-1].strip()
    return result


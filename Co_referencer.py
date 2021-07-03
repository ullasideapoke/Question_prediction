# pip install allennlp==2.1.0 allennlp-models==2.1.0
# pip install transformers

from allennlp.predictors.predictor import Predictor
import time
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import re

import spacy
nlp = spacy.load('en_core_web_sm')

coref_predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')
tf_nlp = pipeline("ner",grouped_entities=True,model='dslim/bert-base-NER')


def tf_ent_detect(sent):
    comp_list = []
    loc_list = []
    person_names = []
    try:
        identified_ents = tf_nlp(sent)
        
        for idx,dt in enumerate(identified_ents):
            if dt['score'] > 0.9 and dt['entity_group'] == 'ORG':
              prob_player = dt['word'].replace("#", "")
              print("prob_player--",prob_player)
              if "#" in dt['word']:
                  wd_cnt = len(dt['word'].split())
                  start_idx = dt['start'] - 2  # (two places for ##)
                  end_idx = dt['end']
                  prob_comp = sent[start_idx:end_idx]
                  print("## mentioned comps-->",prob_comp)
                  if (len(prob_comp.strip().strip("."))>=2) and (prob_comp.split()[0].lower() not in [cm.lower() for comp in comp_list for cm in comp.split()]):
                    print("prob_player1--",prob_player)
                    comp_list.append(prob_comp.strip())

              elif len(dt['word'])<=2:
                wd_cnt = len(dt['word'])
                start_idx = dt['start']
                next_search = start_idx+wd_cnt
                for n_dt in [d for i,d in enumerate(identified_ents) if i!=idx]:
                  if n_dt['word'].startswith("#") and n_dt['start'] == next_search:
                    n_dt_name = n_dt['word'].replace("#","")
                    comp_name = dt['word']+n_dt_name
                    if len(comp_name.strip().strip("."))>=2 and comp_name.split()[0].lower() not in [cm.lower() for comp in comp_list for cm in comp.split()]:
                      print("prob_player2--",prob_player)
                      print("appending after identifying ##-->",comp_name)
                      comp_list.append(comp_name.strip())

              elif len(prob_player.strip().strip(".")) >=2 and prob_player.split()[0].lower() not in [cm.lower() for comp in comp_list for cm in comp.split()]:
                  print("prob_player3--",prob_player)
                  comp_list.append(prob_player.strip())


        for dt in identified_ents:
            if dt['score'] > 0.85 and dt['entity_group'] == 'LOC':
                prob_player = dt['word'].replace("#", "")
                if "#" in dt['word']:
                    start_idx = dt['start'] - 2  # (two places for ##)
                    end_idx = dt['end']
                    prob_loc = sent[start_idx:end_idx]
                    if len(prob_loc)>2:
                      loc_list.append(prob_loc)
                else:
                  if len(prob_player) > 2:
                    loc_list.append(prob_player)

        for dt in identified_ents:
            if dt['score'] > 0.85 and dt['entity_group'] == 'PER':
                prob_player = dt['word'].replace("#", "")
                if "#" in dt['word']:
                    start_idx = dt['start'] - 2  # (two places for ##)
                    end_idx = dt['end']
                    prob_per = sent[start_idx:end_idx]
                    if len(prob_per)>2:
                      person_names.append(prob_per)
                else:
                  if len(prob_player) > 2:
                    person_names.append(prob_player)
    except Exception as ex:
        print("Exception occured in tf_ent_detect-->",ex)
    return list(set(comp_list+loc_list+person_names))



def get_ent_span(doc,tx):
  ent_id_dt = {}
  for token in doc:
    for w in tx.split():
      if w in token.text:
        if w in ent_id_dt.keys():
          ent_id_dt[w].append(token.i)
        else:
          ent_id_dt[w] = [token.i]

  final_wd = tx.split()[-1]
  final_wd_idx = [i for k,v in ent_id_dt.items() if k.strip()==final_wd.strip() for i in v]
  final_wd_least = min(final_wd_idx)

  initial_wd = tx.split()[0]
  initial_wd_idx = [i for k,v in ent_id_dt.items() if k.strip()==initial_wd.strip() for i in v]
  initial_wd_least = min(initial_wd_idx)

  return doc[initial_wd_least:final_wd_least+1]


def core_logic_part(document, coref, resolved, mention_span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def get_span_noun_indices(doc, cluster):
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices

def get_cluster_head(doc, cluster, noun_indices):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    if len(head_span.text.split())>8:
      entity_ls = tf_ent_detect(head_span.text)
      if len(entity_ls) == 1:
        head_span = get_ent_span(doc,entity_ls[0])

      # head_span = entity_ls[0] if len(entity_ls) == 1 else head_span
    return head_span, [head_start, head_end]


def improved_cataphora_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention:  # we don't replace the head itself
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)

def improved_redudant_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:  # if there is at least one noun phrase...
            mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
            mention_span = document[mention_start:mention_end]
            if len(mention_span.text.split())>8:
              entity_ls = tf_ent_detect(mention_span.text)
              if len(entity_ls) == 1:
                mention_span = get_ent_span(document,entity_ls[0])
              for coref in cluster[1:]:
                  core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)



def is_containing_other_spans(span, all_spans):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_nested_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def get_ref_original_clf(content): ### Here topic phrase match should be done
  final_res_lst = []
  final_res_lst_mod = []
  doc = nlp(content)
  clusters = coref_predictor.predict(content)['clusters']
  cataphora = improved_cataphora_replace_corefs(doc, clusters)
  redudant = improved_redudant_replace_corefs(doc, clusters)
  if len(content.split("\n")) == len(cataphora.split("\n")) == len(redudant.split("\n")):
    for idx,sent in enumerate(content.split("\n")):
      original_temp = {}
      modified_sent = []
      if cataphora.split("\n")[idx]!=sent:
        modified_sent.append(cataphora.split("\n")[idx])
      if redudant.split("\n")[idx]!=sent:
        modified_sent.append(redudant.split("\n")[idx])
      if not modified_sent:
        modified_sent.append(sent)

      original_temp['sentence'] = sent
      original_temp['coref_sents'] = list(set(modified_sent))

      final_res_lst.append(original_temp)

  final_res_lst = [dt for dt in final_res_lst if dt['sentence']]
  del clusters,cataphora,redudant


  return final_res_lst

def sent_group_former(content):
    grouped_sent = []
    sent_tokens = sent_tokenize(content)
    finished_ids = []
    for id_,sent in enumerate(sent_tokens):
        if id_ in finished_ids:
            continue
        if len(sent.split()) > 128:
            grouped_sent.append(sent)
            finished_ids.append(id_)
        elif id_==len(sent_tokens)-1:
            finished_ids.append(id_)
            grouped_sent.append(sent)
        else:
            new_sent = sent_tokens[id_]+" "+sent_tokens[id_+1]
            grouped_sent.append(new_sent)
            finished_ids.append(id_)
            finished_ids.append(id_+1)
    return grouped_sent

def group_sent_for_coref(content):
  grouped = []
  split_sent = content.split("\n")
  split_sent = [sent for sent in split_sent if sent]
  if len(split_sent)<4 and len(" ".join(split_sent).split())>150:
    split_sent = sent_group_former(content)
  print("total sents are--->",len(split_sent))
  if len(split_sent)>15:
    start_id = 0
    end_id = 15
    last_ref_sent = ''
    while True:
      temp_list = split_sent[start_id:end_id]
      if last_ref_sent:
        temp_list.append(last_ref_sent)
      temp_str = "\n".join(temp_list)
      coreferenced = get_ref_original_clf(temp_str)
      last_ref = coreferenced[-1]['coref_sents'][0]
      if start_id!=0:
        coreferenced = coreferenced[1:]
      for dt in coreferenced:
        grouped.append(dt)
      if end_id>len(split_sent)-1:
        break

      start_id = end_id
      end_id = end_id+15
      last_ref_sent = last_ref
  else:
      temp_str = "\n".join(split_sent)
      coreferenced = get_ref_original_clf(temp_str)
      for dt in coreferenced:
        grouped.append(dt)

  return grouped


def coreference_pipeline(sentence):
    result = group_sent_for_coref(sentence)
    final_res = result[0].get('coref_sents')
    print("final_res----------", final_res)
    final_res = " ".join(final_res)
    # final_res = text_cleaning(final_res)
    print ("coreference---------",final_res)

    return final_res


if __name__ == '__main__':

    input_text ="Give a breif information about following questions :) What is ABS Technology ?)Where it is used ?)Companies and univeties working on it ?) Market size?"
    output= coreference_pipeline(input_text)
    print(output)



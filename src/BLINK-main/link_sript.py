from blink.main_dense import load_models
from blink.biencoder.biencoder import load_biencoder
from blink.crossencoder.crossencoder import load_crossencoder
from blink.biencoder.data_process import process_mention_data
from tqdm import tqdm
import argparse

def load_candidates(
    entity_catalogue
):
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    id2url = {
    v: "https://en.wikipedia.org/wiki?curid=%s" % k
    for k, v in wikipedia_id2local_id.items()
    }
    return (
        title2id,
        id2title,
        id2text,
        id2url,
        wikipedia_id2local_id,
    )


            
def find_closest_entities(text, top_k_biencoder=100, top_k_crossencoder=20):
    data_to_link = [{"id": 0, "text": text, "mention": text}]
    samples = process_mention_data(data_to_link)
    
    # Biencoder predictions
    _, _, _, _, _, predictions, scores, = biencoder.link_candidate(
        samples, top_k=max(top_k_biencoder, 100), return_scores=True
    )
    
    biencoder_results = []
    for ent_title, score in zip(predictions[0], scores[0]):
        ent_id = title2id[ent_title]
        ent_text = id2text[ent_id]
        ent_url = id2url[ent_id]
        biencoder_results.append({
                    "Entity_ID": int(ent_id),
                    "Entity_Name": ent_title,
                    "Entity_Text": ent_text,
                    "Entity_URL": ent_url,
                    "Score": float(score)
        })
        
    # Crossencoder reranking
    cross_predictions, cross_scores = crossencoder.rerank(
        samples, predictions, scores, top_k=top_k_crossencoder
    )

    cross_results = []
    for ent_title, score in zip(cross_predictions[0], cross_scores[0]):
        ent_id = title2id[ent_title]
        ent_text = id2text[ent_id]
        ent_url = id2url[ent_id]
        cross_results.append({
                    "Entity_ID": int(ent_id),
                    "Entity_Name": ent_title,
                    "Entity_Text": ent_text,
                    "Entity_URL": ent_url,
                    "Score": float(score)
        })

    return biencoder_results, cross_results
            

            
            
path_ent = "/root/autodl-tmp/entity.jsonl"
IPC_path = "/root/autodl-tmp/Sample"
ipc_entities = []

parser = argparse.ArgumentParser()
parser.add_argument('--biencoder_model', type=str, default='models/biencoder_wiki_large.bin')
parser.add_argument('--biencoder_config', type=str, default='models/biencoder_wiki_large.json')
parser.add_argument('--entity_catalogue', type=str, default='models/entity.jsonl')
parser.add_argument('--entity_encoding', type=str, default='models/all_entities_large.t7')
parser.add_argument('--crossencoder_model', type=str, default='models/crossencoder_wiki_large.bin')
parser.add_argument('--crossencoder_config', type=str, default='models/crossencoder_wiki_large.json')
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--fast', action='store_true')
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--show_url', action='store_true')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
parser.add_argument('--lowercase', action='store_true')

args = parser.parse_args()

# 然后，创建配置字典
config = {
    "biencoder_model": args.biencoder_model,
    "biencoder_config": args.biencoder_config,
    "entity_catalogue": args.entity_catalogue,
    "entity_encoding": args.entity_encoding,
    "crossencoder_model": args.crossencoder_model,
    "crossencoder_config": args.crossencoder_config,
    "top_k": args.top_k,
    "output_path": args.output_path,
    "fast": args.fast,
    "interactive": args.interactive,
    "show_url": args.show_url,
    "no_cuda": args.no_cuda,
    "bert_model": args.bert_model,
    "lowercase": args.lowercase
}

biencoder = load_biencoder(config)
crossencoder = load_crossencoder(config)

# 加载实体字典
print("Loading entity dictionary...")
title2id, id2title, id2text, id2url, wikipedia_id2local_id = load_candidates(path_ent)
title_list = list(title2id.keys())
print("Entity dictionary loaded.")

for level in range(3, 4):
    csv_file = f"{IPC_path}/floor_{level}.csv"
    df = pd.read_csv(csv_file)
    print(f"Processing level {level}, total rows: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing level {level}"):
        ipc_classification = row["Classification Identifier"]
        ipc_description = row["Classification Description"]
        row_text = row["Classification Description"]
        extracted_texts = extract_and_process_text(row_text, level)

        texts_entities = []
        for text in extracted_texts:
            biencoder_entities, crossencoder_entities = find_closest_entities(text)

            texts_entities.append({
                "Text": text,
                "Biencoder_Recommended_Entities": biencoder_entities,
                "Separator_mid": "########" * 50,
                "Crossencoder_Recommended_Entities": crossencoder_entities
            })

        ipc_entities.append({
            "Separator_start": "########" * 50,
            "IPC_Classification": ipc_classification,
            "IPC_Description": ipc_description,
            "Texts_Entities": texts_entities,
            "level": level
        })

print("Saving results to file...")
output_file = '/root/autodl-tmp/try_GNN/MAG_dataset/GNN/graph_data/inference_data/linked_level_3.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json_record = json.dumps(ipc_entities, ensure_ascii=False, indent=4)
    file.write(json_record + '\n')

print(f"File saved successfully. File path: {output_file}")

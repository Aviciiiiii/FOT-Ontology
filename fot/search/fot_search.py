from __future__ import annotations

import json
import os
import re
import html
import subprocess
from pathlib import Path
from urllib.parse import quote
from typing import List, Tuple, Dict, Any

from ..utils.logging import get_logger


def clean_html(raw_html: str) -> str:
    """Extract text content from HTML"""
    if not raw_html:
        return ""
    clean_text = re.sub('<.*?>', '', raw_html)
    return html.unescape(clean_text)


def remove_phonetic(text: str) -> str:
    """Remove phonetic annotations"""
    return re.sub(r'\([^)]*\)', '', text)


def clean_text_fn(text: str) -> str:
    """Clean text content"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def google_search(query: str, api_key: str, cse_id: str, proxy: str = None, **kwargs) -> Dict[str, Any]:
    """Execute Google search and return results"""
    logger = get_logger("google_search")
    query_encoded = quote(query)
    params = '&'.join([f'{key}={value}' for key, value in {'q': query_encoded, 'key': api_key, 'cx': cse_id, **kwargs}.items()])
    url = f"https://www.googleapis.com/customsearch/v1?{params}"

    try:
        if proxy:
            # Use SOCKS5 proxy
            curl_command = f"curl -s --socks5-hostname {proxy} -L '{url}'"
            result = subprocess.run(curl_command, shell=True, capture_output=True, text=True, timeout=30)
            if result.stderr:
                logger.warning("Curl stderr for query '%s': %s", query, result.stderr)
            response_text = result.stdout
        else:
            # Direct HTTP request, prefer requests
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except ImportError:
                # If requests not available, fallback to curl
                curl_command = f"curl -s -L '{url}'"
                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True, timeout=30)
                if result.stderr:
                    logger.warning("Curl stderr for query '%s': %s", query, result.stderr)
                response_text = result.stdout

        if response_text:
            return json.loads(response_text)
        else:
            logger.error("Empty response for query: %s", query)
            return {}

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response for query '%s'. Response: %s", query, response_text[:200])
        return {}
    except subprocess.TimeoutExpired:
        logger.error("Timeout occurred for query: %s", query)
        return {}
    except Exception as e:
        logger.error("Error during Google search for query '%s': %s", query, e)
        return {}


def custom_tokenizer(text: str) -> List[str]:
    """Custom tokenizer that handles contractions"""
    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_md')
        except OSError:
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                from spacy.lang.en import English
                nlp = English()

        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.text.strip():  # Ignore whitespace tokens
                if token.text in ["'s", "n't", "'m", "'re", "'ve", "'ll"]:
                    if tokens:  # Ensure there's a previous token
                        tokens[-1] += token.text  # Attach contraction to previous token
                    else:
                        tokens.append(token.text)
                else:
                    tokens.append(token.text)
        return tokens
    except ImportError:
        # Simple tokenization fallback
        return text.split()


def extract_and_tag_sentences(search_results: Dict[str, Any], fot: str) -> List[Dict[str, Any]]:
    """Extract and tag sentences from search results"""
    logger = get_logger("extract_and_tag_sentences")
    dataset = []
    samples_set = set()  # For deduplication
    positive_samples = 0
    negative_samples = 0
    fot_lower = fot.lower()
    fot_words = fot_lower.split()

    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_md')
        except OSError:
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                from spacy.lang.en import English
                nlp = English()
        nlp.add_pipe('sentencizer', last=True)
        use_spacy = True
    except ImportError:
        logger.warning("SpaCy not available, using basic text processing")
        use_spacy = False

    for item in search_results.get('items', []):
        fields = [
            item.get('title', ''),
            item.get('snippet', ''),
            item.get('htmlSnippet', '')
        ]
        if 'pagemap' in item and 'metatags' in item['pagemap']:
            for tag in item['pagemap']['metatags']:
                fields.append(tag.get('og:description', ''))
                fields.append(tag.get('twitter:description', ''))

        for field in fields:
            if not field:
                continue

            clean_text = clean_text_fn(remove_phonetic(clean_html(field)))
            if clean_text not in samples_set:
                samples_set.add(clean_text)

                if use_spacy:
                    doc = nlp(clean_text)
                    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                else:
                    # Basic sentence splitting
                    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]

                for sentence in sentences:
                    tokens = custom_tokenizer(sentence)
                    tags = ['O'] * len(tokens)
                    has_entity = False

                    for i in range(len(tokens)):
                        if ' '.join(tokens[i:]).lower().startswith(fot_lower):
                            tags[i] = 'B-FOT'
                            for j in range(1, len(fot_words)):
                                if i + j < len(tags):
                                    tags[i+j] = 'I-FOT'
                            has_entity = True
                            break  # Stop after finding one match to avoid duplicate tagging

                    dataset.append({"tokens": tokens, "tags": tags})
                    if has_entity:
                        positive_samples += 1
                    else:
                        negative_samples += 1

    logger.info("FOT entity '%s': positive samples: %d, negative samples: %d", fot, positive_samples, negative_samples)
    return dataset


def _tag_sentence(entity: str, text: str) -> dict:
    """Simple fallback sentence tagging for dry-run mode."""
    tokens = text.split()
    labels = ["O"] * len(tokens)
    ent_words = entity.split()
    for i in range(len(tokens) - len(ent_words) + 1):
        if [w.lower() for w in tokens[i : i + len(ent_words)]] == [w.lower() for w in ent_words]:
            labels[i] = "B-FOT"
            for j in range(1, len(ent_words)):
                labels[i + j] = "I-FOT"
            break
    return {"tokens": tokens, "tags": labels, "text": text, "source_id": entity}


def run_fot_search(
    level2_path: str,
    third_entities_path: str,
    out1: str,
    out2: str,
    *,
    dry_run: bool = False,
    fast: bool = True,
    backend: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    """Run FOT search with Google Custom Search API.

    Args:
        level2_path: Path to Level 2 entities JSON file (optional)
        third_entities_path: Path to third entities JSON file
        out1: Output path for first half of results
        out2: Output path for second half of results
        dry_run: If True, generate synthetic data
        fast: If True, limit processing for faster execution
        backend: "google" for real search, "local" for corpus search, "dryrun" for synthetic
        cfg: Configuration dictionary with API keys and settings

    Returns:
        Tuple of output file paths
    """
    logger = get_logger("fot_search")
    cfg = cfg or {}

    # Load entity names from both sources (mimicking original script)
    fot_query = []

    # Load level 2 entities if path provided and file exists
    if level2_path and Path(level2_path).exists():
        try:
            with open(level2_path, 'r', encoding='utf-8') as file:
                onetwo_entities = json.load(file)
            if isinstance(onetwo_entities, list):
                # If it's a list of entities with 'name' field
                if onetwo_entities and isinstance(onetwo_entities[0], dict):
                    onetwo_query = [entity['name'] for entity in onetwo_entities if 'name' in entity]
                else:
                    # If it's already a list of names
                    onetwo_query = onetwo_entities
            else:
                onetwo_query = []
            fot_query.extend(onetwo_query)
            logger.info("Loaded %d Level 2 entities", len(onetwo_query))
        except Exception as e:
            logger.warning("Failed to load Level 2 entities from %s: %s", level2_path, e)

    # Load third entities
    try:
        third_query = json.loads(Path(third_entities_path).read_text(encoding="utf-8"))
        fot_query.extend(third_query)
        logger.info("Loaded %d third entities", len(third_query))
    except Exception as e:
        logger.error("Failed to load third entities from %s: %s", third_entities_path, e)
        return "", ""

    logger.info("Total FOT query entities: %d", len(fot_query))
    samples: List[dict] = []

    # Determine backend mode
    if dry_run or not backend or backend == "dryrun":
        logger.info("Using dry-run mode for FOT search")
        for n in fot_query:
            samples.append(_tag_sentence(n, f"Advances in {n} technology"))
            samples.append(_tag_sentence(n, f"{n} systems and components"))

    elif backend == "google":
        logger.info("Using Google Custom Search API for FOT search")

        # Get Google API configuration
        google_api = cfg.get("google_api", {})
        api_key = google_api.get("fot_api_key") or google_api.get("api_key") or os.getenv("GOOGLE_API_KEY")
        cse_id = google_api.get("fot_cse_id") or google_api.get("cse_id") or os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            logger.error("Google API key or CSE ID not found. Set google_api.fot_api_key and google_api.fot_cse_id in config, or use environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID")
            logger.info("Falling back to dry-run mode")
            for n in fot_query:
                samples.append(_tag_sentence(n, f"Advances in {n} technology"))
                samples.append(_tag_sentence(n, f"{n} systems and components"))
        else:
            # Get proxy configuration
            proxy_cfg = cfg.get("proxy", {})
            proxy = proxy_cfg.get("address") if proxy_cfg.get("enabled", False) else None

            # Get limits
            limits = cfg.get("limits", {})
            max_calls = limits.get("max_api_calls_per_day", 100 if fast else 9980)
            start_index = limits.get("start_index", 0)

            # Limit entities based on fast mode and max_calls
            if fast:
                max_entities = min(len(fot_query), 10, max_calls)
            else:
                max_entities = min(len(fot_query), max_calls)

            end_index = min(start_index + max_entities, len(fot_query))
            query_entities = fot_query[start_index:end_index]

            logger.info("Processing %d entities (from %d to %d)", len(query_entities), start_index, end_index)

            api_calls_count = 0
            for i, entity in enumerate(query_entities):
                if api_calls_count >= max_calls:
                    logger.warning("API call limit reached (%d)", max_calls)
                    break

                logger.info("Processing entity %d/%d: '%s'", i + 1, len(query_entities), entity)

                try:
                    search_results = google_search(entity, api_key, cse_id, proxy)
                    if search_results:
                        tagged_data = extract_and_tag_sentences(search_results, entity)
                        samples.extend(tagged_data)
                        logger.info("Retrieved %d samples for entity: %s", len(tagged_data), entity)
                    else:
                        logger.warning("No search results for entity: %s", entity)
                        # Add fallback sample
                        samples.append(_tag_sentence(entity, f"Basics of {entity} design"))

                    api_calls_count += 1

                except Exception as e:
                    logger.error("Error processing entity '%s': %s", entity, e)
                    # Add fallback sample
                    samples.append(_tag_sentence(entity, f"Advances in {entity} technology"))
                    api_calls_count += 1

    else:  # backend == "local"
        logger.info("Using local corpus search for FOT search")
        topk = int(cfg.get("topk", 10))
        nt = Path("files/15.FieldsOfStudy.nt")
        corpus_lines: List[str] = []
        if nt.exists():
            try:
                with nt.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= (200 if fast else 2000):
                            break
                        corpus_lines.append(line.strip())
            except Exception:
                corpus_lines = []

        for n in fot_query:
            found = 0
            lower = n.lower()
            for ln in corpus_lines:
                if lower in ln.lower():
                    txt = ln if len(ln.split()) > 4 else f"Study of {n} applications"
                    samples.append(_tag_sentence(n, txt))
                    found += 1
                    if found >= max(1, topk // 5):
                        break
            if found == 0:
                samples.append(_tag_sentence(n, f"Basics of {n} design"))

    # Split samples into two files
    half = (len(samples) + 1) // 2
    out_p1 = Path(out1)
    out_p2 = Path(out2)
    out_p1.parent.mkdir(parents=True, exist_ok=True)
    out_p2.parent.mkdir(parents=True, exist_ok=True)

    # Write output files with same format as original (indent=4)
    out_p1.write_text(json.dumps(samples[:half], ensure_ascii=False, indent=4), encoding="utf-8")
    out_p2.write_text(json.dumps(samples[half:], ensure_ascii=False, indent=4), encoding="utf-8")

    logger.info("FOT search completed: %d samples -> [%s (%d), %s (%d)]",
                len(samples), out_p1, len(samples[:half]), out_p2, len(samples[half:]))
    return str(out_p1), str(out_p2)

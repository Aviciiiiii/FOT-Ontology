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
    """Removes HTML tags from a string and unescapes HTML entities."""
    if not raw_html:
        return ""
    clean_text = re.sub('<.*?>', '', raw_html)
    return html.unescape(clean_text)


def google_search(query: str, api_key: str, cse_id: str, proxy: str = None, **kwargs) -> Dict[str, Any]:
    """Execute a Google Custom Search and return the JSON result."""
    logger = get_logger("google_search")
    query_encoded = quote(query)
    params = '&'.join([f'{key}={value}' for key, value in {'q': query_encoded, 'key': api_key, 'cx': cse_id, **kwargs}.items()])
    url = f"https://www.googleapis.com/customsearch/v1?{params}"

    try:
        if proxy:
            # Using SOCKS5 proxy as specified in the original script
            curl_command = f"curl -s --socks5-hostname {proxy} -L '{url}'"
            result = subprocess.run(curl_command, shell=True, capture_output=True, text=True, timeout=30)
            if result.stderr:
                logger.warning("Curl stderr for query '%s': %s", query, result.stderr)
            response_text = result.stdout
        else:
            # Direct HTTP request using requests if available, fallback to curl
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except ImportError:
                # Fallback to curl if requests not available
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


def extract_and_tag_sentences(search_results: Dict[str, Any], entity_to_tag: str) -> List[Dict[str, Any]]:
    """Extract sentences from search results and tag the specified entity."""
    logger = get_logger("extract_and_tag_sentences")
    samples_set = set()
    dataset = []

    try:
        # Try to load spaCy for better text processing
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            try:
                nlp = spacy.load('en_core_web_md')
            except OSError:
                # Fallback to basic English model
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
            item.get('htmlTitle', ''),
            item.get('snippet', ''),
            item.get('htmlSnippet', '')
        ]

        # Extract metadata fields
        if 'pagemap' in item and 'metatags' in item['pagemap']:
            for tag in item['pagemap']['metatags']:
                fields.append(tag.get('og:description', ''))
                fields.append(tag.get('twitter:description', ''))

        for field in fields:
            if not field:
                continue

            clean_text = clean_html(field)
            # Normalize whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if clean_text and clean_text not in samples_set:
                samples_set.add(clean_text)

                if use_spacy:
                    # Use spaCy for sentence segmentation
                    doc = nlp(clean_text)
                    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                else:
                    # Basic sentence splitting
                    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]

                # Tag sentences containing the entity
                entity_words = entity_to_tag.lower().split()
                for sentence in sentences:
                    if use_spacy:
                        doc = nlp(sentence)
                        words = [token.text for token in doc]
                    else:
                        words = sentence.split()

                    tags = ['O'] * len(words)
                    has_entity = False

                    # Sliding window to find the entity phrase
                    for i in range(len(words) - len(entity_words) + 1):
                        # Compare lowercased words to ensure match
                        if [word.lower() for word in words[i:i + len(entity_words)]] == entity_words:
                            tags[i] = 'B-FOT'  # Begin tag
                            for j in range(1, len(entity_words)):
                                tags[i + j] = 'I-FOT'  # Inside tag
                            has_entity = True
                            break  # Stop after finding the first match in the sentence

                    dataset.append({"tokens": words, "tags": tags})

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


def run_mag_search(
    entities_path: str,
    out1: str,
    out2: str,
    *,
    dry_run: bool = False,
    fast: bool = True,
    backend: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    """Run MAG search with Google Custom Search API.

    Args:
        entities_path: Path to JSON file containing entity names
        out1: Output path for first half of results
        out2: Output path for second half of results
        dry_run: If True, generate synthetic data
        fast: If True, limit processing for faster execution
        backend: "google" for real search, "local" for corpus search, "dryrun" for synthetic
        cfg: Configuration dictionary with API keys and settings

    Returns:
        Tuple of output file paths
    """
    logger = get_logger("mag_search")
    names: List[str] = json.loads(Path(entities_path).read_text(encoding="utf-8"))
    samples: List[dict] = []
    cfg = cfg or {}

    # Determine backend mode
    if dry_run or not backend or backend == "dryrun":
        logger.info("Using dry-run mode for MAG search")
        for n in names:
            samples.append(_tag_sentence(n, f"Research on {n} methods"))
            samples.append(_tag_sentence(n, f"Applications of {n} in industry"))

    elif backend == "google":
        logger.info("Using Google Custom Search API for MAG search")

        # Get Google API configuration
        google_api = cfg.get("google_api", {})
        api_key = google_api.get("mag_api_key") or google_api.get("api_key") or os.getenv("GOOGLE_API_KEY")
        cse_id = google_api.get("mag_cse_id") or google_api.get("cse_id") or os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            logger.error("Google API key or CSE ID not found. Set google_api.mag_api_key and google_api.mag_cse_id in config, or use environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID")
            logger.info("Falling back to dry-run mode")
            for n in names:
                samples.append(_tag_sentence(n, f"Research on {n} methods"))
                samples.append(_tag_sentence(n, f"Applications of {n} in industry"))
        else:
            # Get proxy configuration
            proxy_cfg = cfg.get("proxy", {})
            proxy = proxy_cfg.get("address") if proxy_cfg.get("enabled", False) else None

            # Get limits
            limits = cfg.get("limits", {})
            max_calls = limits.get("max_api_calls_per_day", 100 if fast else 1000)
            start_index = limits.get("start_index", 0)

            # Limit entities based on fast mode and max_calls
            if fast:
                max_entities = min(len(names), 10, max_calls)
            else:
                max_entities = min(len(names), max_calls)

            end_index = min(start_index + max_entities, len(names))
            query_entities = names[start_index:end_index]

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
                        samples.append(_tag_sentence(entity, f"Overview of {entity} in practice"))

                    api_calls_count += 1

                except Exception as e:
                    logger.error("Error processing entity '%s': %s", entity, e)
                    # Add fallback sample
                    samples.append(_tag_sentence(entity, f"Study of {entity} concepts"))
                    api_calls_count += 1

    else:  # backend == "local"
        logger.info("Using local corpus search for MAG search")
        topk = int(cfg.get("topk", 10))
        # Local backend: read a tiny slice from files/15.FieldsOfStudy.nt if exists
        corpus_lines: List[str] = []
        nt = Path("files/15.FieldsOfStudy.nt")
        if nt.exists():
            try:
                with nt.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= (200 if fast else 2000):
                            break
                        corpus_lines.append(line.strip())
            except Exception:
                corpus_lines = []

        # Simple retrieval: take lines containing entity terms (case-insensitive)
        for n in names:
            found = 0
            lower = n.lower()
            for ln in corpus_lines:
                if lower in ln.lower():
                    txt = ln if len(ln.split()) > 4 else f"Study of {n} concepts"
                    samples.append(_tag_sentence(n, txt))
                    found += 1
                    if found >= max(1, topk // 5):
                        break
            if found == 0:
                # Fallback sentence if nothing matched
                samples.append(_tag_sentence(n, f"Overview of {n} in practice"))

    # Split samples into two files
    half = (len(samples) + 1) // 2
    out_p1 = Path(out1)
    out_p2 = Path(out2)
    out_p1.parent.mkdir(parents=True, exist_ok=True)
    out_p2.parent.mkdir(parents=True, exist_ok=True)

    # Write output files with same format as original (indent=4)
    out_p1.write_text(json.dumps(samples[:half], ensure_ascii=False, indent=4), encoding="utf-8")
    out_p2.write_text(json.dumps(samples[half:], ensure_ascii=False, indent=4), encoding="utf-8")

    logger.info("MAG search completed: %d samples -> [%s (%d), %s (%d)]",
                len(samples), out_p1, len(samples[:half]), out_p2, len(samples[half:]))
    return str(out_p1), str(out_p2)

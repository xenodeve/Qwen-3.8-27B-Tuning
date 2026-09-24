"""Normalize work counts without crediting reused tokens as fresh prefill."""


def prefill_work(response):
    timings = response.get('timings') or {}
    usage = response.get('usage') or {}
    total = usage.get('prompt_tokens')
    cached = timings.get('cache_n', timings.get('cached_tokens'))
    if cached is None:
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    result = {'engine_prefill_tps': timings.get('prompt_per_second'),
              'prefill_ms': timings.get('prompt_ms'), 'cached_tokens': cached,
              'uncached_prompt_tokens': None, 'uncached_prefill_tps': None}
    if total is None or cached is None:
        result['prefill_count_status'] = 'unknown'
        return result
    if not 0 <= cached <= total:
        result['prefill_count_status'] = 'invalid-cache-count'
        return result
    new = total - cached
    result['uncached_prompt_tokens'] = new
    result['prefill_count_status'] = 'cache-only' if new == 0 else 'measured-counts'
    if new and timings.get('prompt_ms', 0) > 0:
        result['uncached_prefill_tps'] = new * 1000 / timings['prompt_ms']
    return result

from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists:list[list[any]], k: float = 60.0):     
    score_map = defaultdict(float)

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            score_map[doc] += 1.0 / (k + rank)
    
    # Sort by descending fused score
    sorted_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
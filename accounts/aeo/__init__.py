"""
AEO prompt engine (prompt definitions + generation utilities).

Example (inside a view or service):

    from accounts.models import BusinessProfile
    from accounts.aeo import (
        aeo_business_input_from_profile,
        combine_prompt_set,
        generate_dynamic_prompts,
        generate_fixed_prompts,
        run_prompt_batch_via_openai,
    )

    profile = BusinessProfile.objects.get(pk=profile_id)
    ctx = aeo_business_input_from_profile(
        profile,
        city="Salt Lake City",
        services=["same day crowns", "sedation dentistry"],
        niche_modifiers=["sedation"],
    )
    prompts = combine_prompt_set(
        generate_fixed_prompts(ctx.industry, ctx.city),
        generate_dynamic_prompts(ctx),
    )
    # Optional LLM expansion:
    # more = run_prompt_batch_via_openai(ctx, seed_prompts=prompts)
    # prompts = combine_prompt_set(prompts, more)

    # Phase 2 — execute and store raw answers (one OpenAI call per prompt):
    # from accounts.aeo import run_aeo_prompt_batch
    # run_aeo_prompt_batch(prompts, profile, save=True)

    # Phase 3 — structured extraction from stored raw rows:
    # from accounts.aeo import run_single_extraction
    # for snap in profile.aeo_response_snapshots.order_by("-created_at")[:20]:
    #     run_single_extraction(snap, save=True)

    # Phase 4 — score from latest extractions (append-only snapshots):
    # from accounts.aeo import calculate_aeo_scores_for_business
    # calculate_aeo_scores_for_business(profile, save=True)

    # Phase 5 — recommendations (stores AEORecommendationRun when save=True):
    # from accounts.aeo import generate_aeo_recommendations
    # generate_aeo_recommendations(profile, save=True)
"""

from .aeo_execution_utils import (
    hash_prompt,
    normalize_prompt_for_hash,
    run_aeo_prompt_batch,
    run_single_aeo_prompt,
    save_aeo_response,
)
from .aeo_recommendation_utils import (
    analyze_citation_gaps,
    analyze_visibility_gaps,
    generate_aeo_recommendations,
    generate_natural_language_recommendation,
    save_recommendation_run,
)
from .aeo_scoring_utils import (
    calculate_aeo_scores_for_business,
    calculate_citation_share,
    calculate_competitor_dominance,
    calculate_visibility_score,
    calculate_weighted_position_score,
    latest_extraction_per_response,
    save_aeo_score_snapshot,
)
from .aeo_extraction_utils import (
    extract_aeo_response,
    normalize_extraction_payload,
    root_domain_from_fragment,
    run_single_extraction,
    save_extraction_result,
)
from .aeo_utils import (
    AEOPromptBusinessInput,
    aeo_business_input_from_profile,
    build_full_aeo_prompt_plan,
    build_openai_batch_user_content,
    combine_prompt_set,
    generate_dynamic_prompts,
    generate_fixed_prompts,
    infer_city_from_address,
    normalize_aeo_prompt_dict,
    parse_aeo_prompt_json_array,
    prepare_structured_extraction_input,
    run_prompt_batch_via_openai,
)
from .aeo_prompts import (
    AEOPromptTemplateSpec,
    AEOPromptType,
    AEO_EXECUTION_SYSTEM_PROMPT,
    AEO_EXTRACTION_PREP_SYSTEM_PROMPT,
    AEO_PROMPT_ENGINE_SYSTEM_PROMPT,
    AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE,
    AEO_RECOMMENDATION_NL_SYSTEM_PROMPT,
    GENERIC_COMPETITOR_TOKENS,
)

__all__ = [
    "AEOPromptBusinessInput",
    "AEOPromptTemplateSpec",
    "AEOPromptType",
    "AEO_EXECUTION_SYSTEM_PROMPT",
    "AEO_EXTRACTION_PREP_SYSTEM_PROMPT",
    "AEO_PROMPT_ENGINE_SYSTEM_PROMPT",
    "AEO_RECOMMENDATION_NL_SYSTEM_PROMPT",
    "AEO_STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "AEO_STRUCTURED_EXTRACTION_USER_TEMPLATE",
    "GENERIC_COMPETITOR_TOKENS",
    "aeo_business_input_from_profile",
    "analyze_citation_gaps",
    "analyze_visibility_gaps",
    "build_full_aeo_prompt_plan",
    "build_openai_batch_user_content",
    "calculate_aeo_scores_for_business",
    "calculate_citation_share",
    "calculate_competitor_dominance",
    "calculate_visibility_score",
    "calculate_weighted_position_score",
    "combine_prompt_set",
    "extract_aeo_response",
    "generate_aeo_recommendations",
    "generate_natural_language_recommendation",
    "generate_dynamic_prompts",
    "generate_fixed_prompts",
    "hash_prompt",
    "infer_city_from_address",
    "latest_extraction_per_response",
    "normalize_aeo_prompt_dict",
    "normalize_extraction_payload",
    "normalize_prompt_for_hash",
    "parse_aeo_prompt_json_array",
    "prepare_structured_extraction_input",
    "root_domain_from_fragment",
    "run_aeo_prompt_batch",
    "run_prompt_batch_via_openai",
    "run_single_aeo_prompt",
    "run_single_extraction",
    "save_aeo_response",
    "save_aeo_score_snapshot",
    "save_extraction_result",
    "save_recommendation_run",
]

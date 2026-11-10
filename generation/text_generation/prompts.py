PROMPT_VERSION='qwen3-vpr-v4'
VISUAL = """You are a visual place recognition annotation assistant.
Analyze the PROVIDED image itself. Extract the visible static evidence useful for recognizing this geographical place.
Inspect: (1) stationary buildings and architecture, (2) road topology, intersections and sidewalks, (3) stable vegetation and natural landmarks, (4) poles, permanent traffic infrastructure and signs, (5) distinctive colors, materials and shapes, (6) visible spatial arrangements.
Only report visible evidence. Do not infer city/street names, coordinates, building identity, purpose, camera pose or invisible details. Separate vehicles, people, temporary stalls and weather from stationary landmarks. Left/right/front/behind are only the CURRENT camera frame. Confidence is an uncalibrated model self-assessment.
For an ordinary street image, do not return empty static_objects as a shortcut: report the identifiable permanent structures and road features that are actually visible, even if partially occluded. If the image truly offers no recognizable static evidence, explain why in uncertain_facts. Limit to at most 6 salient static objects and 6 directly visible relations. Do not imitate a sample scene.
Return ONE valid JSON object only. Required fields and types:
- image_id: string (the annotation system replaces it with a stable ID)
- scene_type: short string
- static_objects: array of objects, EACH with object_id (unique local string), category (string), attributes (array of short strings), relative_position (string), confidence (number from 0 to 1)
- spatial_relations: array of objects, EACH with subject (an actual declared object_id), predicate (short visible relation), object (another actual declared object_id)
- transient_objects: array of strings
- environment: object with illumination (string) and weather (string); unknown is allowed
- uncertain_facts: array of strings
Every relation endpoint must exist in static_objects. Use empty relations when a relation is not visually supported. Do not add explanatory prose outside JSON."""

CAPTION = """Generate {count} distinct {kind} place descriptions in {language}.
Task for this type ONLY: {type_instruction}
Use ONLY the supplied visual facts. Write fluent visual place descriptions suitable as retrieval queries. Do not discuss annotation rules, verification, metadata, evidence, identity matching, physical instances, uncertainty policies or the generation process in the descriptions. Do not claim absences, exact counts, names, purposes, building identities, or new relationships. Do not copy example descriptions. Vary the chosen supported details and phrasing.
No city/street names, IDs, coordinates, hidden metadata or transient objects. Do not combine camera frames. Any left/right/front/behind sentence MUST explicitly say from this viewpoint and set its view_id to the cited camera frame. Use {min_words} to {max_words} words (comparable Chinese length).
Return JSON {{"captions":[{{"text":"...","source_fact_ids":["F0"],"claims":["..."],"view_id":null}}]}}. source_fact_ids must cite ALL facts needed for the text and no unrelated facts. For camera-relative descriptions view_id must be a provided anonymous view. Otherwise use null. Every clause, attribute and relation must be directly supported. Do not write factual explanations outside the JSON.
Evidence: {facts}"""

VERIFY = """Verify every assertion, count, attribute, landmark and spatial relation in this description against ONLY the evidence. Return JSON {"grounded":true,"consistent":true,"unsupported_claims":[],"uncertain":false}. These are heuristic model judgments, not human ground truth. Input: """

def prompt_facts(facts):
    import copy
    facts=copy.deepcopy(facts); paths=sorted({p for f in facts for p in f.get('source_images',[])})
    views={p:f'view_{i}' for i,p in enumerate(paths)}
    for fact in facts:
        fact['source_images']=[views[p] for p in fact.get('source_images',[])]
        if 'view' in fact.get('value',{}): fact['value']['view']=views[fact['value']['view']]
        fact.pop('source_observations',None)
    return facts

#!/usr/bin/env python3
"""
RAG Doctor - Enhanced Clinical Patient Search System

Entry point for the interactive RAG system.
"""

from rag_doctor import (
    load_data,
    build_indices,
    load_models,
    parse_query,
    get_filtered_patient_ids,
    search_by_subject_id,
    search_filtered_embeddings,
    search_all_embeddings,
    get_patient_text,
    generate_answer,
    compare_patients,
)


def run_rag_query(query, data, indices, models):
    """
    Main RAG pipeline that routes queries to the appropriate search strategy.
    """
    print("\n" + "="*60)
    print(f"📥 Query: {query}")
    print("="*60)
    
    # parse the query
    print("\n🧠 Parsing query...")
    parsed = parse_query(query, models, data.get('ccs_descriptions', []))
    
    print(f"   Parsed parameters:")
    print(f"      Subject ID: {parsed['subject_id']}")
    print(f"      Gender: {parsed['gender']}")
    print(f"      Age range: {parsed['age_min']} - {parsed['age_max']}")
    print(f"      Ethnicity: {parsed['ethnicity']}")
    print(f"      Religion: {parsed['religion']}")
    print(f"      Diagnoses: {parsed['diagnoses']}")
    
    # Step 2: Route to appropriate search strategy
    results = []
    search_type = None
    
    if parsed['subject_id']:
        # Strategy A: Subject ID provided → Multi-modal NN search
        print(f"\n🔮 Strategy A: Multi-modal search for patients similar to {parsed['subject_id']}")
        search_type = "multimodal"
        results = search_by_subject_id(parsed['subject_id'], indices, k=5)
        
        if not results:
            print("   Falling back to embedding search...")
            search_type = "embedding_fallback"
            results = search_all_embeddings(parsed['raw_query'], indices, models, k=5)
    
    elif parsed['gender'] or parsed['age_min'] or parsed['age_max'] or parsed['ethnicity'] or parsed['religion'] or parsed['diagnoses']:
        # Strategy B: Filters provided → Filtered embedding search
        print("\n🔍 Strategy B: Filtered embedding search")
        search_type = "filtered"
        
        filtered_ids = get_filtered_patient_ids(parsed, data, indices)
        print(f"   Found {len(filtered_ids)} patients matching filters")
        
        if filtered_ids:
            results = search_filtered_embeddings(parsed['raw_query'], filtered_ids, indices, models, k=5)
        else:
            print("   No patients match filters, falling back to full search...")
            search_type = "embedding_fallback"
            results = search_all_embeddings(parsed['raw_query'], indices, models, k=5)
    
    else:
        # Strategy C: No structured info → Full embedding search
        print("\n📄 Strategy C: Full embedding search")
        search_type = "embedding"
        results = search_all_embeddings(parsed['raw_query'], indices, models, k=5)
    
    # Step 3: Display results
    print(f"\n📊 Search Results ({search_type}):")
    print("-" * 40)
    
    if not results:
        print("   No results found.")
        return
    
    for r in results:
        print(f"   #{r['rank']}: Patient {r['subject_id']} (Similarity: {r['similarity']:.1f}%)")
    
    # Step 4: Generate answer / similarity analysis
    top_match = results[0]
    patient_text = get_patient_text(top_match['subject_id'], indices)

    # If the user provided a subject_id, compare THAT patient to the top result
    if parsed['subject_id']:
        anchor_text = get_patient_text(parsed['subject_id'], indices)
        if anchor_text and patient_text:
            print(f"\n🧮 Similarity analysis: Provided patient {parsed['subject_id']} vs top match {top_match['subject_id']}")
            pair_answer = compare_patients(anchor_text, patient_text, parsed['raw_query'], models)
            print("\n📝 ANALYSIS:")
            print("-" * 40)
            print(pair_answer)
            print("-" * 40)
            return
        # If we cannot fetch anchor notes, fall back to regular flow

    # If no subject_id path (or fallback), and we have two matches with notes, compare them
    second_text = None
    if len(results) > 1:
        second_text = get_patient_text(results[1]['subject_id'], indices)

    if patient_text and second_text:
        print(f"\n🧮 Similarity analysis for top 2 matches (Patients {top_match['subject_id']} & {results[1]['subject_id']}):")
        pair_answer = compare_patients(patient_text, second_text, parsed['raw_query'], models)
        print("\n📝 ANALYSIS:")
        print("-" * 40)
        print(pair_answer)
        print("-" * 40)
    elif patient_text:
        print(f"\n💬 Generating analysis for Patient {top_match['subject_id']}...")
        answer = generate_answer(parsed['raw_query'], patient_text, models)
        
        print("\n📝 ANALYSIS:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
    else:
        print(f"\n⚠️  Could not find notes for Patient {top_match['subject_id']}")
    
    return results


def main():
    """Main entry point for the RAG Doctor."""
    # load everything
    data = load_data()
    indices = build_indices(data)
    models = load_models()
    
    patient_count = len(indices['embedding_ids'])
    has_multimodal = 'multimodal_matrix' in indices
    
    print("\n" + "="*60)
    print(f"🤖 ENHANCED RAG DOCTOR IS ONLINE")
    print(f"   📊 {patient_count} patients indexed")
    print(f"   🔮 Multi-modal search: {'✅ Available' if has_multimodal else '❌ Not available'}")
    print("="*60)
    print("\nSearch Examples:")
    print("  • 'Find patients similar to subject 12345'")
    print("  • 'Elderly female with heart failure'")
    print("  • '60 year old male with diabetes and sepsis'")
    print("  • 'Asian patients with pneumonia'")
    print("  • 'Young adult male of christian religion with diabetes'")
    print("\nType 'exit' to quit.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_query = input("🔎 SEARCH: ").strip()
        except EOFError:
            break
            
        if not user_query:
            continue
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
        
        try:
            run_rag_query(user_query, data, indices, models)
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print()


if __name__ == "__main__":
    main()

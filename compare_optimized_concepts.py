import json
import numpy as np
import torch
import os

def load_optimized_concepts(file_path):
    """Load optimized concepts from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_concept_data():
    """Load original concept data"""
    concept_raw = np.load('data/concepts.npy', allow_pickle=True)
    concept2cls = np.load('data/concept2cls.npy')
    return concept_raw, concept2cls

def create_concept_class_mapping(concept_raw, concept2cls):
    """Create mapping from concept to its original class"""
    concept_to_class = {}
    for i, concept in enumerate(concept_raw):
        class_id = concept2cls[i]
        class_name = "engaging" if class_id == 1 else "not_engaging"
        concept_to_class[concept] = class_name
    return concept_to_class

def compare_concepts(optimized_concepts, concept_to_class):
    """Compare optimized concepts with original concept2cls mapping"""
    results = {
        "not_engaging": {
            "correct": 0,
            "total": 0,
            "correct_concepts": []
        },
        "engaging": {
            "correct": 0,
            "total": 0,
            "correct_concepts": []
        }
    }
    
    # Check each class
    for class_name in ["not_engaging", "engaging"]:
        class_concepts = optimized_concepts.get(class_name, [])
        results[class_name]["total"] = len(class_concepts)
        
        # Check each concept
        for concept in class_concepts:
            if concept in concept_to_class:
                original_class = concept_to_class[concept]
                if original_class == class_name:
                    results[class_name]["correct"] += 1
                    results[class_name]["correct_concepts"].append(concept)
    
    return results

def print_statistics(results):
    """Print statistics about the comparison"""
    total_correct = results["not_engaging"]["correct"] + results["engaging"]["correct"]
    total_concepts = results["not_engaging"]["total"] + results["engaging"]["total"]
    
    print("=== COMPARISON RESULTS ===")
    print(f"Total concepts in optimized_concepts.json: {total_concepts}")
    
    if total_concepts > 0:
        accuracy = total_correct / total_concepts * 100
        print(f"Overall accuracy: {accuracy:.2f}% ({total_correct}/{total_concepts})")
    
    for class_name in ["not_engaging", "engaging"]:
        class_result = results[class_name]
        class_total = class_result["total"]
        class_correct = class_result["correct"]
        
        print(f"\n{class_name.upper()} CLASS:")
        print(f"Total concepts: {class_total}")
        
        if class_total > 0:
            class_accuracy = class_correct / class_total * 100
            print(f"Accuracy: {class_accuracy:.2f}% ({class_correct}/{class_total})")
        
        print("\nFirst 10 correctly classified concepts:")
        for concept in class_result["correct_concepts"][:10]:
            print(f"  - {concept}")
    
    print("\n=== SUMMARY ===")
    print(f"not_engaging: {results['not_engaging']['correct']}/{results['not_engaging']['total']} correct")
    print(f"engaging: {results['engaging']['correct']}/{results['engaging']['total']} correct")

def main():
    """Main function to compare optimized concepts with concept2cls"""
    # Check if files exist
    if not os.path.exists("results/optimized_concepts.json"):
        print("Error: results/optimized_concepts.json not found")
        return
    
    if not os.path.exists("data/concepts.npy") or not os.path.exists("data/concept2cls.npy"):
        print("Error: concept data files not found in data/ directory")
        return
    
    # Load data
    optimized_concepts = load_optimized_concepts("results/optimized_concepts.json")
    concept_raw, concept2cls = load_concept_data()
    
    # Create mapping
    concept_to_class = create_concept_class_mapping(concept_raw, concept2cls)
    
    # Compare
    results = compare_concepts(optimized_concepts, concept_to_class)
    
    # Print statistics
    print_statistics(results)

if __name__ == "__main__":
    main() 
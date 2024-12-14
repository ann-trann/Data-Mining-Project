import pandas as pd
import numpy as np
from itertools import combinations

def perform_rough_set_analysis(df):
    # Separate Information System and Decision System
    columns = df.columns
    decision_column = columns[-1]  # Last column is the decision attribute
    condition_columns = list(columns[:-1])
    
    # Information System
    information_system = df[condition_columns].values
    decision_system = df[decision_column].values
    
    # Indiscernibility Relation
    def indiscernibility_relation(data):
        """Compute indiscernibility classes"""
        indiscernibility_classes = {}
        for i, row in enumerate(data):
            row_key = tuple(row)
            if row_key not in indiscernibility_classes:
                indiscernibility_classes[row_key] = []
            indiscernibility_classes[row_key].append(i)
        return list(indiscernibility_classes.values())
    
    # Indiscernibility Relation for Condition and Decision Attributes
    condition_indiscernibility = indiscernibility_relation(information_system)
    decision_indiscernibility = indiscernibility_relation(decision_system.reshape(-1, 1))
    
    # Set Approximation
    def set_approximation(condition_classes, target_classes):
        """Compute lower and upper approximations"""
        lower_approx = []
        upper_approx = []
        
        for target_class in target_classes:
            lower_class = []
            upper_class = []
            
            for condition_class in condition_classes:
                if all(idx in target_class for idx in condition_class):
                    lower_class.extend(condition_class)
                if any(idx in target_class for idx in condition_class):
                    upper_class.extend(condition_class)
            
            lower_approx.append(set(lower_class))
            upper_approx.append(set(upper_class))
        
        return lower_approx, upper_approx
    
    # Compute Approximations
    lower_approx, upper_approx = set_approximation(condition_indiscernibility, decision_indiscernibility)
    
    # Rough Set Quality (Accuracy)
    def rough_set_quality(lower_approx, upper_approx):
        """Compute rough set approximation quality"""
        qualities = []
        for lower, upper in zip(lower_approx, upper_approx):
            if len(upper) > 0:
                quality = len(lower) / len(upper)
                qualities.append(quality)
        return np.mean(qualities) if qualities else 0
    
    accuracy = rough_set_quality(lower_approx, upper_approx)
    
    # Attribute Dependency
    def attribute_dependency(condition_columns, decision_column, df):
        """Compute dependency of condition attributes to decision attribute"""
        dependencies = {}
        for k in range(1, len(condition_columns) + 1):
            for subset in combinations(condition_columns, k):
                subset_columns = list(subset)
                subset_df = df[subset_columns + [decision_column]]
                
                # Compute dependency using entropy or other methods
                # This is a simplified version
                unique_subset = subset_df.drop_duplicates()
                dependency_score = len(unique_subset[unique_subset.columns[-1]].unique()) / len(df[decision_column].unique())
                
                dependencies[', '.join(subset)] = dependency_score
        
        return dependencies
    
    # Reduct (Minimal subset of attributes that preserves discernibility)
    def find_reduct(condition_columns, decision_column, df):
        """Find minimal subset of attributes that preserves discernibility"""
        dependencies = attribute_dependency(condition_columns, decision_column, df)
        
        # Sort dependencies and select top attributes
        sorted_deps = sorted(dependencies.items(), key=lambda x: x[1], reverse=True)
        
        # Select attributes that maximize dependency
        reduct = []
        max_dependency = 0
        for attrs, score in sorted_deps:
            candidate_reduct = reduct + attrs.split(', ')
            candidate_df = df[candidate_reduct + [decision_column]]
            
            # Check if adding these attributes increases dependency
            if score > max_dependency:
                reduct = candidate_reduct
                max_dependency = score
        
        return list(set(reduct))
    
    # Compute Reduct
    reduct = find_reduct(condition_columns, decision_column, df)
    
    return {
        "information_system": condition_columns,
        "decision_system": decision_column,
        "condition_indiscernibility": len(condition_indiscernibility),
        "decision_indiscernibility": len(decision_indiscernibility),
        "lower_approximation_size": len(lower_approx),
        "upper_approximation_size": len(upper_approx),
        "approximation_accuracy": accuracy,
        "attribute_dependencies": dict(attribute_dependency(condition_columns, decision_column, df)),
        "reduct": reduct
    }





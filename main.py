import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import fuzzy_expert as fz

from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.inference import DecompositionalInference

# Define fuzzy variables with sorted terms
variables_K = {
    "Kalium": FuzzyVariable(
        universe_range=(0, 100), terms={
            'Low': [(0,1), (20, 1), (30, 0)],
            'Medium': [(20, 0), (30, 1), (40, 1), (50,0)],
            'High': [(40, 0), (50, 1)]
        }
    ),
    "KCl": FuzzyVariable(
        universe_range=(0, 300), terms={
            'Low': [(0,1), (50, 1), (100, 0)],
            'High': [(50, 0),(100, 1)]
        }
    )
}

variables_N = {
    "Nitrogen": FuzzyVariable(
        universe_range=(0, 100), terms={
            'Low': [(0.2, 1), (0.3, 0)],
            'Medium': [(0.2, 0), (0.3, 1), (0.5,1) ,(0.6, 0)],
            'High': [(0.5, 0), (0.6, 1)]
        }
    ),
    "Urea": FuzzyVariable(
        universe_range=(0, 300), terms={
            'Low': [(27, 1), (116, 0)],
            'Medium': [(27, 0), (116, 1), (161, 0)],
            'High': [(116, 0), (123, 0.2), (136, 0.5), (150, 0.8), (161, 1)]
        }
    )
}

variables_P = {
    "Potassium": FuzzyVariable(
        universe_range=(0, 100), terms={
            'Low': [(20, 1), (30, 0)],
            'Medium': [(20, 0), (30,1), (40, 1), (50,0)],
            'High': [(40, 0),  (50, 1)]
        }
    ),
    "SP36": FuzzyVariable(
        universe_range=(0, 300), terms={
            'Low': [(50, 1),  (75, 0)],
            'Medium': [(50, 0), (75, 1),(100, 0)],
            'High': [(75, 0),  (100, 1)]
        }
    )
}

# Define fuzzy rules
rules_N = [
    FuzzyRule(
        premise=[("Nitrogen", "Low")],
        consequence=[("Urea", "High")]
    ),
    FuzzyRule(
        premise=[("Nitrogen", "Medium")],
        consequence=[("Urea", "Medium")]
    ),
    FuzzyRule(
        premise=[("Nitrogen", "High")],
        consequence=[("Urea", "Low")]
    ),
]

rules_K = [
    FuzzyRule(
        premise=[("Kalium", "Low")],
        consequence=[("KCl", "High")]
    ),
    FuzzyRule(
        premise=[("Kalium", "Medium")],
        consequence=[("KCl", "Low")]
    ),
    FuzzyRule(
        premise=[("Kalium", "High")],
        consequence=[("KCl", "Low")]
    ),
]

rules_P = [
    FuzzyRule(
        premise=[("Potassium", "Low")],
        consequence=[("SP36", "High")]
    ),
    FuzzyRule(
        premise=[("Potassium", "Medium")],
        consequence=[("SP36", "Medium")]
    ),
    FuzzyRule(
        premise=[("Potassium", "High")],
        consequence=[("SP36", "Low")]
    ),
]

# Define the fuzzy inference model
model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator="Rc",
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog"
)

# Get user input for Kalium, Nitrogen, and Potassium levels
kalium_input = float(input("Enter the Kalium level in soil (0.0-100): "))
nitrogen_input = float(input("Enter the Nitrogen level in soil (0.0 - 100): "))
potassium_input = float(input("Enter the Potassium level in soil (0.0-100): "))

# Perform inference for variables_N (Nitrogen and Urea)
result_N, infered_cf_N = model(
    variables=variables_N,
    rules=rules_N,
    Nitrogen=nitrogen_input,
)

# Print the defuzzified inferred memberships for Nitrogen and Urea
print("Inference results for Nitrogen and Urea:")
for key, value in result_N.items():
    print(f"{key}: {value}")

# Print the inferred confidence level for Nitrogen and Urea
print(f"Inferred Confidence Level (Nitrogen and Urea): {infered_cf_N}")

# Perform inference for variables_K (Kalium and KCl)
result_K, infered_cf_K = model(
    variables=variables_K,
    rules=rules_K,
    Kalium=kalium_input,
)

# Print the defuzzified inferred memberships for Kalium and KCl
print("\nInference results for Kalium and KCl:")
for key, value in result_K.items():
    print(f"{key}: {value}")

# Print the inferred confidence level for Kalium and KCl
print(f"Inferred Confidence Level (Kalium and KCl): {infered_cf_K}")

# Perform inference for variables_P (Potassium and SP36)
result_P, infered_cf_P = model(
    variables=variables_P,
    rules=rules_P,
    Potassium=potassium_input,
)

# Print the defuzzified inferred memberships for Potassium and SP36
print("\nInference results for Potassium and SP36:")
for key, value in result_P.items():
    print(f"{key}: {value}")

# Print the inferred confidence level for Potassium and SP36
print(f"Inferred Confidence Level (Potassium and SP36): {infered_cf_P}")




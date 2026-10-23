
# Research Questions

## Working Title

Beyond Accuracy: Evaluating the Stability of Feature Attributions
in Tabular Machine Learning

## Main Research Question

How stable are feature attribution explanations under controlled,
realistic input perturbations when the predictive model remains fixed?

## Subquestions

1. How does attribution stability change as perturbation strength increases?
2. Do different tabular model families exhibit different stability patterns?
3. How strongly is explanation stability associated with predictive
   performance changes?
4. Do ranking-based and value-based stability metrics lead to
   different conclusions?

## Initial Hypothesis

Small input perturbations may produce substantial changes in feature
attributions even when the model's predictions change only slightly.

This hypothesis must be evaluated empirically.

## Scope

- Tabular supervised learning.
- Fixed trained model during the primary perturbation experiment.
- Controlled perturbations of input features.
- Feature attribution methods, initially SHAP.
- Separate evaluation of prediction changes and explanation changes.

## Research Integrity

- No fabricated results.
- No claims of novelty before reviewing related literature.
- All experiments must be reproducible.
- Dataset provenance and limitations must be documented.

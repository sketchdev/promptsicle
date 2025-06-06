# Promptsicle

A DX friendly library for standardizing and optimizing prompts for development projects.

## Ideas

- Create a function that takes a data loader, optimizer, scoring (metric) functions etc. and returns an optimized prompt or generated set of prompts.
- Maybe have default / precanned data loaders and optimizers.
- Maybe have a folder loader that loads examples from a folder. Maybe it should also have a file that represents the last score as the target to beat.`
- Maybe have an default / precanned openai scoring function that returns a score based on the output of the model.
- Maybe have a default / precanned human scoring function that shows a result and asks the user to score it.
- Maybe it should generate a file with functions for each prompt (inspired by the supabase types generator).
- Input parameters to the generated function should be fixed so that don't break contracts.
- This could enable standardization of prompt generation, optimization, and usage for developers.
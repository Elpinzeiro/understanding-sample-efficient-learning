"""factdepth: code of "From one example to pretraining: how supervision moves the depth at which facts become readable
in LLMs" (UniReps 2026).

Modules, in the order a run uses them:

    config       models, J-lenses, LoRA sites, grids (sizes, steps, learning rate), result folders
    io           logging, atomic JSON files
    data         the frozen dataset: one Unit per (language pair, relation); prompts, training sentences
    matching     does an answer name the city (English / Chinese / Russian rules)
    model        model loading, greedy generation
    lens         the J-lens readout
    llm          OpenRouter client (the judge's GPT-4o-mini)
    judge        the answer judge (version 5; version 1 for the learning-rate sweep behind Figure 2)
    evaluate     one checkpoint: answers, readout, judge, known facts after, forgetting
    inject       one LoRA injection: training, evaluation at every checkpoint
    run          the GPU entry point of the injections: python -m factdepth.run experiments/configs/<x>.yaml
    lens_refit   App. C.4: J-lenses refitted on the untouched and the fine-tuned model, their readout and agreement
    known_ft     App. C.6: the known facts read with each saved adapter on
    analysis/    every figure, table and number of the paper (metrics, plots, tables, numbers)
"""

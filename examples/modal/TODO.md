## todos
- write a spec that avoids excessive abstraction, but exposes the training benchmark pipeline steps conveniently
  - train single run (just pass w/ kwargs)
  - sweeps: probably roll this out into a separate file, esp. if abstraction is necessary/more convenient here
- dataset regen (and maybe preprocessing) currently uses a single node, but should work w/ autoscaling. can do this by forking/rolling our own regen script
- using sglang for online training, e.g. figure out how/where to place the engines in a node/cluster
- regen code, server side, should handle entire conversation. scrap any mention of sticky session/flash container routing on server side
- later: dataset stuff
  - should work on a generic specdec pretraining dataset. should include perfectblend but also subset of toolcalling dataset. Salesforce/xlam-function-calling-60k is a good one.
  - dataset size -> should be motivated by model capacity i.e. param count of draft model. will differ per-architecture + per-model family, hence dataset should accommodate the max variant of these
- later: draft model arch is generally llama-based, but the configuration is based on the target model config (specifically head dim, num heads, etc.) (def suboptimal!)
- later: runtime-configurable GPUs
- later: DDP training w/ @clustered
- later: may need to tweak the token-wise exponential decay for opd-loss. it's currently fixed at 0.8 for the entire loss. options:
  1. just tweak the coarse decay, or
  2. add a parameter to modulate the decay _just_ for the opd-related parts of the loss

## big picture
- how to demonstrate improved performance of OPD draft model?
- hypothesis: OPD draft model should be more robust to slight/medium changes in data
  - ways to measure robustness:
    - train on one dataset, eval on others
    - add a new tool
    - change existing tool call behavior
  - lit review state of open evals for this
- must show some benefit while also showing similar training speed/metrics/overall cost

Yes — TRL already supports VLM GRPO, not just LLMs. Their GRPO docs explicitly say GRPO supports VLM training on multimodal datasets with text + images, tested with Gemma3, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, and SmolVLM2. They also provide a grpo_vlm.py example and automatically handle image / images columns via the model processor.

For HOPPR/vlm_align: yes, I would strongly consider migrating toward HF TRL if your model can be made TRL-compatible. The question is not “does TRL support VLMs?” It does. The question is whether your private VLM architecture fits TRL’s assumptions.

I’d frame it this way:

Use TRL if:

the model can be represented as a normal HF PreTrainedModel / processor stack;
image inputs can be passed as image / images columns;
you can LoRA-target the language decoder or projection layers through PEFT;
you do not need custom rollout internals or vision caching.

Keep custom vlm_align trainer if:

your model has nonstandard fields like images_pixels / images_masks;
you rely on custom vision-cache behavior across rollouts;
you need monkey patches or private model-loading logic;
TRL cannot cleanly express “freeze vision encoder / QFormer, adapt only decoder” for your architecture.

My recommendation: do a spike, not a rewrite. Try to reproduce one small vlm_align GRPO experiment using TRL’s GRPOTrainer + your RadEval-style reward. If you can get parity on inputs, LoRA targeting, reward logging, and throughput, migrate. If not, keep your trainer but align its reward interface with TRL/RadEval so the two worlds remain compatible.

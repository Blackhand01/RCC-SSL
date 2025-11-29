# TODO: Apply patches to fix xai pipeline bugs

- [ ] Replace entire ask_concept method in vlm_client.py with the provided patch
- [ ] Update test_llava_call in debug_llava.py to use /worker_generate and stop="###"
- [ ] Fix selection in config_concept.yaml: remove lowconf_topk from per_class, add global_low_conf.topk
- [ ] Fix selection in config_xai.yaml: same change
- [ ] Verify changes applied correctly

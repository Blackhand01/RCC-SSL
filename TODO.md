# TODO for updating run_pa-llava_pipeline.py to use venv

- [x] Define ROOT = Path(__file__).resolve().parent.parent
- [x] Update run_cmd function to def run_cmd(label, cmd, cwd=None, env=None):
- [x] Update convert_domain_alignment: change cmd1 and cmd2 to use [sys.executable, "-m", "xtuner", ...] and update run_cmd calls with labels
- [x] Update convert_instruction_tuning similarly
- [ ] Update run_pathvqa: change cmd to use [sys.executable, "-m", "xtuner", ...] and update run_cmd call with label
- [ ] Update run_pmcvqa similarly
- [ ] Update run_zero_shot similarly
- [ ] Remove call to ensure_xtuner_available() in main()
- [ ] Remove ensure_xtuner_available function

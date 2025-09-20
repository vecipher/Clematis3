# Scheduler examples

## Round-robin (visible rotation)
```bash
python3 scripts/run_demo.py \
  --config examples/scheduler/round_robin_minimal.yaml \
  --agents AgentA,AgentB,AgentC --steps 6